import logging
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase

from torsk.models.initialize import (
    dense_esn_reservoir, scale_weight, sparse_esn_reservoir)
import torsk.models.torch_optimize as opt

logger = logging.getLogger(__name__)


class TorchESNCell(RNNCellBase):
    """An Echo State Network (ESN) cell.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of features in the hidden state
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
    input : torch.Tensor
        contains input features of shape (batch, input_size)
    state : torch.Tensor
        current hidden state of shape (batch, hidden_size)

    Outputs
    -------
    state' : torch.Tensor
        contains the next hidden state of shape (batch, hidden_size)
    """
    def __init__(
            self, input_size, hidden_size,
            spectral_radius, in_weight_init, in_bias_init, density, dtype):
        super(TorchESNCell, self).__init__(
            input_size, hidden_size, bias=True, num_chunks=1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = getattr(torch, dtype)

        in_weight = torch.rand([hidden_size, input_size], dtype=self.dtype)
        in_weight = scale_weight(in_weight, in_weight_init)
        self.in_weight = Parameter(in_weight, requires_grad=False)

        weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.weight_hh = Parameter(
            torch.tensor(weight_hh, dtype=self.dtype), requires_grad=False)

        in_bias = torch.rand([hidden_size], dtype=self.dtype)
        in_bias = scale_weight(in_bias, in_bias_init)
        self.in_bias = Parameter(in_bias, requires_grad=False)
        self.res_bias = self.register_parameter('res_bias', None)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def forward(self, inputs, state):
        self.check_forward_input(inputs)
        self.check_forward_hidden(inputs, state)
        self.check_dtypes(inputs, state)
        return torch._C._VariableFunctions.rnn_tanh_cell(
            inputs, state, self.in_weight, self.weight_hh,
            self.in_bias, self.res_bias)


class SparseTorchESNCell(RNNCellBase):
    """An Echo State Network (ESN) cell with a sparsely represented reservoir
    matrix. Can currently only handle batch==1.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of features in the hidden state
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
    input : torch.Tensor
        contains input features of shape (batch, input_size)
    state : torch.Tensor
        current hidden state of shape (batch, hidden_size)

    Outputs
    -------
    state' : torch.Tensor
        contains the next hidden state of shape (batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size,
                 spectral_radius, in_weight_init, in_bias_init, density):
        super(SparseTorchESNCell, self).__init__(
            input_size, hidden_size, bias=True, num_chunks=1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = getattr(torch, params.dtype)

        # input matrix
        in_weight = torch.rand([hidden_size, input_size])
        in_weight = scale_weight(in_weight, in_weight_init)
        self.weight_ih = Parameter(in_weight, requires_grad=False)

        # sparse reservoir matrix
        matrix = sparse_esn_reservoir(
            dim=hidden_size,
            spectral_radius=spectral_radius,
            density=density,
            symmetric=False)

        matrix = matrix.tocoo()
        indices = torch.LongTensor([matrix.row, matrix.col])
        values = torch.FloatTensor(matrix.data)

        self.weight_hh = Parameter(torch.sparse.FloatTensor(
            indices, values, [hidden_size, hidden_size]), requires_grad=False)

        # biases
        in_bias = torch.rand([hidden_size, 1])
        in_bias = scale_weight(in_bias, in_bias_init)
        self.bias_ih = Parameter(torch.Tensor(in_bias), requires_grad=False)
        self.bias_hh = self.register_parameter('bias_hh', None)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def forward(self, inputs, state):
        if not inputs.size(0) == state.size(0) == 1:
            raise ValueError("SparseTorchESNCell can only process batch_size==1")
        self.check_forward_input(inputs)
        self.check_forward_hidden(inputs, state)
        self.check_dtypes(inputs, state)

        # reshape for matrix multiplication
        inputs = inputs.reshape([-1, 1])
        state = state.reshape([-1, 1])

        # next state
        x_inputs = torch.mm(self.weight_ih, inputs)
        x_state = torch.mm(self.weight_hh, state)
        new_state = torch.tanh(x_inputs + x_state + self.bias_ih)

        return new_state.reshape([1, -1])


class TorchESN(nn.Module):
    """Complete ESN with output layer. Only supports batch=1 for now!!!

    Parameters
    ----------
    params : torsk.utils.Params
        The network hyper-parameters

    Inputs
    ------
    inputs : Tensor
        Inputs of shape (seq, batch, input_size)
    state : Tensor
        Initial state of the ESN with shape (batch, hidden_size)
    nr_predictions : int
        Number of steps to predict into the future

    Outputs
    -------
    outputs : Tensor
        Predicitons nr_predictions into the future
        shape (seq, batch, input_size)
    states' : Tensor
        Accumulated states of the ESN with shape (seq, batch, hidden_size)
    """
    def __init__(self, params):
        super(TorchESN, self).__init__()
        self.params = params
        if params.input_size != params.input_size:
            raise ValueError(
                "Currently input and output dimensions must be the same.")

        if params.reservoir_representation == "dense":
            ESNCell = TorchESNCell
        elif params.reservoir_representation == "sparse":
            ESNCell = SparseTorchESNCell

        self.esn_cell = ESNCell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init,
            in_bias_init=params.in_bias_init,
            density=params.density,
            dtype=params.dtype)

        torch.set_default_dtype(self.esn_cell.dtype)

        self.out = nn.Linear(
            params.hidden_size + params.input_size + 1,
            params.input_size,
            bias=False)

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
        inp = initial_inputs
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
            A batch of labels with shape (batch, input_size)
        """
        if len(inputs.size()) == 3:
            inputs = inputs.reshape([-1, inputs.size(2)])
            states = states.reshape([-1, states.size(2)])
            labels = labels.reshape([-1, labels.size(2)])

        method = self.params.train_method
        beta = self.params.tikhonov_beta

        if method == 'tikhonov':
            if beta is None:
                raise ValueError(
                    'For Tikhonov training the beta parameter cannot be None.')
            wout = opt.tikhonov(inputs, states, labels, beta)

        elif method == 'pinv':
            if beta is not None:
                logger.debug('With pseudo inverse training the '
                             'beta parameter has no effect.')
            wout = opt.pseudo_inverse(inputs, states, labels)

        else:
            raise ValueError(f'Unkown training method: {method}')

        if(wout.shape != self.out.weight.shape):
            raise ValueError("Optimized and original Wout shape do not match."
                             f"{wout.shape} / {self.out.weight.shape}")
        self.out.weight = Parameter(wout, requires_grad=False)
