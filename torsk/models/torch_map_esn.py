import logging
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase
import torchvision.transforms.functional as ttf

from torsk.data.conv import get_kernel as get_np_kernel, conv2d_output_shape
from torsk.models.initialize import dense_esn_reservoir, sparse_nzpr_esn_reservoir

logger = logging.getLogger(__name__)


def get_kernel(kernel_shape, kernel_type, dtype):
    kernel = get_np_kernel(kernel_shape, kernel_shape)
    return torch.tensor(kernel, dtype=dtype)


def input_map(image, input_map_specs):
    features = []
    for spec in input_map_specs:
        if spec["type"] == "pixels":
            _features = ttf.resize(image, spec["size"]).reshape(-1)
            _features = _features * spec["input_scale"]
        elif spec["type"] == "dct":
            raise NotImplementedError
        elif spec["type"] == "conv":
            _features = F.conv2d(image, spec["kernel"]).reshape(-1)
            _features = _features * spec["input_scale"]
        elif spec["type"] == "random_weights":
            _features = torch.mv(spec["weight_ih"], image.reshape(-1))
        else:
            raise ValueError(spec)
        features.append(_features)
    return torch.cat(features, dim=0)


def init_input_map_specs(input_map_specs, input_shape, dtype):
    for spec in input_map_specs:
        if spec["type"] == "conv":
            spec["kernel"] = get_kernel(spec["size"], spec["kernel_type"], dtype)
        elif spec["type"] == "random_weights":
            spec["weight_ih"] = torch.rand(
                spec["size"], input_shape[0] * input_shape[1], dtype=getattr(torch, dtype))
    return input_map_specs


def get_hidden_size(input_shape, input_map_specs):
    hidden_size = 0
    for spec in input_map_specs:
        if spec["type"] == "conv":
            _shape = conv2d_output_shape(input_shape, spec["size"])
            hidden_size += _shape[0] * _shape[1]
        elif spec["type"] == "random_weights":
            hidden_size = spec["size"]
        else:
            shape = spec["size"]
            hidden_size += shape[0] * shape[1]
    return hidden_size


class TorchMapESNCell(RNNCellBase):
    """An Echo State Network (ESN) cell that enables custom input -> state
    mappings.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images
    input_map_specs : list
        a list of dicts that specify the input mapping. example:
        [{"type": "random_weights", "size": 500, "weight_scale": 1.0},
         {"type": "pixels", "size": [10, 10]},
         {"type": "dct", "size": [10, 10]},
         {"type": "conv", "size": [2, 2], "kernel_type": "gauss"}]
    spectral_radius : float
        Largest eigenvalue of the reservoir matrix
    in_weight_init : float
        Input matrix will be chosen from a random uniform like
        (-in_weight_init, in_weight_init)
    in_bias_init : float
        Input matrix will be chosen from a random uniform like
        (-in_bias_init, in_bias_init)


    Inputs
    ------
    input : array
        contains input features of shape (batch, input_size)
    state : array
        current hidden state of shape (batch, hidden_size)

    Outputs
    -------
    state' : array
        contains the next hidden state of shape (batch, hidden_size)
    """

    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype):
        self.input_shape = input_shape
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = getattr(torch, dtype)
        self.input_map_specs = input_map_specs

        input_size = input_shape[0] * input_shape[1]
        hidden_size = self.get_hidden_size(input_shape)
        super(TorchMapESNCell, self).__init__(
            input_size, hidden_size, bias=False, num_chunks=1)
        logger.info(f"ESN hidden size: {self.hidden_size}")

        self.weight_ih = Parameter(
            torch.eye(self.hidden_size, dtype=self.dtype), requires_grad=False)

        weight_hh = dense_esn_reservoir(
            dim=self.hidden_size, spectral_radius=self.spectral_radius,
            density=self.density, symmetric=False)
        self.weight_hh = Parameter(
            torch.tensor(weight_hh, dtype=self.dtype), requires_grad=False)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, self.dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        if not image.size() == torch.Size([1,] + self.input_shape):
            raise ValueError(
                f"Input image shape {image.size()} differs from expected"
                f"{self.input_shape}")
        features = input_map(image, self.input_map_specs)
        return features[None, :]

    def forward(self, image, state):
        inputs = self.input_map(image)
        self.check_forward_hidden(inputs, state)
        self.check_dtypes(inputs, state)
        return torch._C._VariableFunctions.rnn_tanh_cell(
            inputs, state, self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh)


class TorchMapSparseESNCell(RNNCellBase):
    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype):
        self.input_shape = input_shape
        self.input_map_specs = input_map_specs
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = getattr(torch, dtype)

        input_size = input_shape[0] * input_shape[1]
        hidden_size = self.get_hidden_size(input_shape)
        super(TorchMapSparseESNCell, self).__init__(
            input_size, hidden_size, bias=False, num_chunks=1)
        logger.info(f"ESN hidden size: {self.hidden_size}")

        self.weight_ih = Parameter(
            torch.eye(self.hidden_size, dtype=self.dtype), requires_grad=False)

        nonzeros_per_row = int(self.hidden_size * density)
        weight_hh = sparse_nzpr_esn_reservoir(
            dim=self.hidden_size,
            spectral_radius=self.spectral_radius,
            nonzeros_per_row=nonzeros_per_row,
            dtype=dtype)
        indices = torch.LongTensor([weight_hh.row_idx, weight_hh.col_idx])
        values = torch.tensor(weight_hh.values, dtype=self.dtype)
        self.weight_hh = Parameter(torch.sparse.FloatTensor(
            indices, values, [hidden_size, hidden_size]),
            requires_grad=False)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        if not image.size() == torch.Size([1,] + self.input_shape):
            raise ValueError(
                f"Input image shape {image.size()} differs from expected"
                f"{self.input_shape}")
        features = input_map(image, self.input_map_specs)
        return features[None, :]

    def forward(self, image, state):
        x_inputs = self.input_map(image)

        self.check_dtypes(x_inputs, state)
        self.check_forward_hidden(x_inputs, state)

        x_inputs = x_inputs.reshape([-1, 1])
        state = state.reshape([-1, 1])

        x_state = torch.mm(self.weight_hh, state)
        new_state = torch.tanh(x_inputs + x_state)

        return new_state.reshape([1, -1])
