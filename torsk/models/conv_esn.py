import logging

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase

from torsk.models.utils import sparse_esn_reservoir
from torsk.models.esn import tikhonov, pseudo_inverse

logger = logging.getLogger(__name__)


def _mean_kernel(kernel_size):
    size = [kernel_size, kernel_size]
    return torch.ones(size) / (kernel_size * kernel_size)


def _random_kernel(kernel_size):
    logger.warning(f"Random kernel created with hard coded uniform dist")
    size = [kernel_size, kernel_size]
    return torch.rand(size) * 2. - 1.


def get_filter(kernel_size, kernel_type):
    # TODO: gaussian kernel
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_size)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_size)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel.unsqueeze(0).unsqueeze(0)


def _out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    num = in_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return num / stride + 1


class ConvESNCell(RNNCellBase):
    def __init__(self, input_shape, kernels, spectral_radius, density):
        self.hidden_size = self.get_hidden_size(input_shape, kernels)
        super(ConvESNCell, self).__init__(
            0, self.hidden_size, bias=False, num_chunks=1)

        self.input_shape = input_shape

        # input matrix
        self.filters = [get_filter(**kernel) for kernel in kernels]
        # self.weight_ih = Parameter(self.filters, requires_grad=False)

        # sparse reservoir matrix
        matrix = sparse_esn_reservoir(
            dim=self.hidden_size,
            spectral_radius=spectral_radius,
            density=density,
            symmetric=False)

        matrix = matrix.tocoo()
        indices = torch.LongTensor([matrix.row, matrix.col])
        values = torch.FloatTensor(matrix.data)

        self.weight_hh = Parameter(torch.sparse.FloatTensor(
            indices, values, [self.hidden_size] * 2), requires_grad=False)

        # biases
        self.bias_ih = self.register_parameter('bias_ih', None)
        self.bias_hh = self.register_parameter('bias_hh', None)

    def forward(self, inputs, state):
        if not inputs.size(0) == state.size(0) == 1:
            raise ValueError("SparseESNCell can only process batch_size==1")

        inputs = inputs.reshape([1, 1, inputs.size(1), inputs.size(2)])
        state = state.reshape([-1, 1])

        x_inputs = []
        for filt in self.filters:
            conv = F.conv2d(inputs, filt)
            x_inputs.append(conv.reshape(-1))
        x_inputs = torch.cat(x_inputs, dim=0).unsqueeze(-1)

        x_state = torch.mm(self.weight_hh, state)
        new_state = torch.tanh(x_inputs + x_state)

        return new_state.reshape([1, -1])

    def get_hidden_size(self, input_shape, kernels):
        size = 0
        for kernel in kernels:
            height = _out_size(input_shape[0], kernel["kernel_size"])
            width = _out_size(input_shape[1], kernel["kernel_size"])
            size += height * width
        return int(size)


class ConvESN(torch.nn.Module):
    def __init__(self, params):
        super(ConvESN, self).__init__()
        self.params = params

        self.esn_cell = ConvESNCell(
            input_shape=params.input_shape,
            kernels=params.kernels,
            spectral_radius=params.spectral_radius,
            density=params.density)

        ydim, xdim = params.input_shape
        input_size = ydim * xdim

        self.out = torch.nn.Linear(
            self.esn_cell.hidden_size + input_size + 1,
            input_size, bias=False)

        self.ones = torch.ones([1, 1])

    def forward(self, inputs, state, states_only=True):
        if inputs.size(1) != 1:
            raise ValueError("Supports only batch size of one -.-")
        if states_only:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state)

    def _forward_states_only(self, inputs, state):
        states = []
        for inp in inputs:
            state = self.esn_cell(inp, state)
            states.append(state)
        return None, torch.stack(states, dim=0)

    def _forward(self, inputs, state):
        outputs, states = [], []
        for inp in inputs:
            state = self.esn_cell(inp, state)
            ext_state = torch.cat([self.ones, inp, state], dim=1)
            output = self.out(ext_state)
            outputs.append(output)
            states.append(state)
        return torch.stack(outputs, dim=0), torch.stack(states, dim=0)

    def predict(self, initial_inputs, initial_state, nr_predictions):
        inp = initial_inputs.reshape([1, -1, 1])
        state = initial_state
        outputs, states = [], []

        for ii in range(nr_predictions):
            state = self.esn_cell(inp, state)
            ext_state = torch.cat([self.ones, inp, state], dim=1)
            output = self.out(ext_state)
            inp = output

            outputs.append(output)
            states.append(state)
        return torch.stack(outputs, dim=0), torch.stack(states, dim=0)

    def optimize(self, inputs, states, labels):
        """Train the output layer.

        Parameters
        ----------
        inputs : Tensor
            A batch of inputs with shape (batch, input_size)
        states : Tensor
            A batch of hidden states with shape (batch, hidden_size)
        labels : Tensor
            A batch of labels with shape (batch, output_size)
        """
        inputs = inputs.reshape([inputs.size(0), -1])
        states = states.reshape([states.size(0), -1])
        labels = labels.reshape([labels.size(0), -1])

        method = self.params.train_method
        beta = self.params.tikhonov_beta

        if method == 'tikhonov':
            if beta is None:
                raise ValueError(
                    "For Tikhonov training the beta parameter cannot be None.")
            wout = tikhonov(inputs, states, labels, beta)

        elif method == 'pinv':
            if beta is not None:
                logger.warning("With pseudo inverse training the "
                               "beta parameter has no effect.")
            wout = pseudo_inverse(inputs, states, labels)

        else:
            raise ValueError(f"Unkown training method: {method}")

        if(wout.size() != self.out.weight.size()):
            raise ValueError(
                f"Wout shape: {wout.shape} "
                "is not as expected: {self.out.weight.shape}")

        self.out.weight = Parameter(wout, requires_grad=False)
