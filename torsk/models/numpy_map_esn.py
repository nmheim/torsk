import logging
import numpy as np
from scipy.signal import convolve2d

from torsk.data.conv import get_kernel, conv2d_output_shape
from torsk.data.utils import resample2d, normalize
from torsk.data.dct import dct2
from torsk.models.initialize import dense_esn_reservoir, sparse_nzpr_esn_reservoir

logger = logging.getLogger(__name__)


def input_map(image, input_map_specs):
    features = []
    for spec in input_map_specs:
        if spec["type"] == "pixels":
            _features = resample2d(image, spec["size"]).reshape(-1)
        elif spec["type"] == "dct":
            _features = dct2(image, *spec["size"]).reshape(-1)
        elif spec["type"] == "gradient":
            _features = normalize(
                np.concatenate(np.gradient(image)).reshape(-1)) * 2 - 1
        elif spec["type"] == "conv":
            _features = convolve2d(
                image, spec["kernel"], mode='same', boundary="symm").reshape(-1)
            _features = normalize(_features) * 2 - 1
        elif spec["type"] == "random_weights":
            _features = np.dot(spec["weight_ih"], image.reshape(-1))
        else:
            raise ValueError(spec)
        _features = spec["input_scale"] * _features
        features.append(_features)
    return features


def init_input_map_specs(input_map_specs, input_shape, dtype):
    for spec in input_map_specs:
        if spec["type"] == "conv":
            spec["kernel"] = get_kernel(spec["size"], spec["kernel_type"], dtype)
        elif spec["type"] == "random_weights":
            assert len(spec["size"]) == 1
            weight_ih = np.random.uniform(low=-1., high=1.,
                size=[spec["size"][0], input_shape[0] * input_shape[1]])
            spec["weight_ih"] = weight_ih.astype(dtype)
    return input_map_specs


def get_hidden_size(input_shape, input_map_specs):
    hidden_size = 0
    for spec in input_map_specs:
        if spec["type"] == "conv":
            if spec["mode"] == "valid":
                shape = conv2d_output_shape(input_shape, spec["size"])
            elif spec["mode"] == "same":
                shape = input_shape
                hidden_size += shape[0] * shape[1]
        elif spec["type"] == "random_weights":
            hidden_size += spec["size"][0]
        elif spec["type"] == "gradient":
            hidden_size += input_shape[0] * input_shape[1] * 2  # For 2d pictures
        else:
            shape = spec["size"]
            hidden_size += shape[0] * shape[1]
    return hidden_size


class NumpyMapESNCell(object):
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
        self.dtype = np.dtype(dtype)
        self.input_map_specs = input_map_specs

        self.hidden_size = self.get_hidden_size(input_shape)
        logger.info(f"ESN hidden size: {self.hidden_size}")
        self.weight_hh = dense_esn_reservoir(
            dim=self.hidden_size, spectral_radius=self.spectral_radius,
            density=self.density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(self.dtype)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        return input_map(image, self.input_map_specs)

    def cat_input_map(self, input_stack):
        return np.concatenate(input_stack, axis=0)

    def state_map(self, state):
        return np.dot(self.weight_hh, state)

    def forward(self, image, state):
        self.check_dtypes(image, state)

        input_stack = self.input_map(image)
        x_input = self.cat_input_map(input_stack)
        x_state = self.state_map(state)
        new_state = np.tanh(x_input + x_state)

        return new_state


class NumpyMapSparseESNCell(object):
    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype):
        self.input_shape = input_shape
        self.input_map_specs = input_map_specs
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = np.dtype(dtype)

        self.hidden_size = self.get_hidden_size(input_shape)
        logger.info(f"ESN hidden size: {self.hidden_size}")
        nonzeros_per_row = int(self.hidden_size * density)
        self.weight_hh = sparse_nzpr_esn_reservoir(
            dim=self.hidden_size,
            spectral_radius=self.spectral_radius,
            nonzeros_per_row=nonzeros_per_row,
            dtype=self.dtype)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        return input_map(image, self.input_map_specs)

    def cat_input_map(self, input_stack):
        return np.concatenate(input_stack, axis=0)

    def state_map(self, state):
        return self.weight_hh.sparse_dense_mv(state)

    def forward(self, image, state):
        self.check_dtypes(image, state)

        input_stack = self.input_map(image)
        x_input = self.cat_input_map(input_stack)
        x_state = self.state_map(state)
        new_state = np.tanh(x_input + x_state)

        return new_state
