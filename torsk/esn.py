import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase
from torsk.utils import Params


_module_dir = pathlib.Path(__file__).absolute().parent


def get_default_params():
   return Params(_module_dir / 'default_esn_params.json')


def connection_mask(dim, density, symmetric):
    """Creates a square mask with a given density of ones"""
    mask = np.random.uniform(low=0., high=1., size=(dim, dim)) < density
    if symmetric:
        triu = np.triu(mask, k=1)
        tril = np.tril(mask.T)
        mask = triu + tril
    return mask


def dense_esn_reservoir(dim, spectral_radius, density, symmetric):
    """Creates a dense square matrix with random non-zero elements according
    to the density parameter and a given spectral radius.
    
    Parameters
    ----------
    dim : int
        specifies the dimensions of the square matrix
    spectral_radius : float
        largest eigenvalue of the created matrix
    symmetric : bool
        defines if the created matrix is symmetrix or not

    Returns
    -------
    np.ndarray
        square reservoir matrix
    """
    mask = connection_mask(dim, density, symmetric)
    res = np.random.normal(loc=0.0, scale=1.0, size=[dim, dim])
    if symmetric:
        res = np.triu(res) + np.tril(res.T, k=-1)
    res *= mask.astype(float)
    if spectral_radius:
        eig = np.linalg.eigvals(res)
        rho = np.abs(eig).max()
        res = spectral_radius * res / rho
    return res


class ESNCell(RNNCellBase):
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
            spectral_radius, in_weight_init, density):
        super(ESNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.spectra_radius = spectral_radius
        self.in_weight_init = in_weight_init

        in_weight = torch.rand([hidden_size, input_size])
        in_weight *= 2. * in_weight_init
        in_weight -= in_weight_init
        self.in_weight = Parameter(in_weight, requires_grad=False)

        res_weight = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.res_weight = Parameter(
            torch.tensor(res_weight, dtype=torch.float32), requires_grad=False)

        # TODO deal with biases
        self.in_bias = self.register_parameter('in_bias', None)
        self.res_bias = self.register_parameter('res_bias', None)

    def forward(self, inputs, state):
        self.check_forward_input(inputs)
        self.check_forward_hidden(inputs, state)
        return self._backend.RNNTanhCell(
            inputs, state, self.in_weight, self.res_weight,
            self.in_bias, self.res_bias)


class ESN(nn.Module):
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
        shape (seq, batch, output_size)
    states' : Tensor
        Accumulated states of the ESN with shape (seq, batch, hidden_size)
    """
    def __init__(self, params):
        super(ESN, self).__init__()
        if params.input_size != params.output_size:
            raise ValueError(
                "Currently input and output dimensions must be the same.")
        self.esn_cell = ESNCell(params.input_size, params.hidden_size,
            params.spectral_radius, params.in_weight_init, params.density)
        self.out = nn.Linear(params.hidden_size, params.output_size, bias=False)

    def forward(self, inputs, state, nr_predictions=0):
        if inputs.size(1) != 1:
            raise ValueError("Supports only batch size of one -.-")
        if nr_predictions == 0:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state, nr_predictions)

    def _forward_states_only(self, inputs, state):
        states = []
        for inp in inputs:
            state = self.esn_cell(inp, state)
            states.append(state)
        return None, torch.stack(states, dim=0)

    def _forward(self, inputs, state, nr_predictions):
        outputs = []
        for inp in inputs:
            state = self.esn_cell(inp, state)
            output = self.out(state)
            outputs.append(output)
        for ii in range(nr_predictions):
            inp = output
            state = self.esn_cell(inp, state)
            output = self.out(state)
            outputs.append(output)
        return torch.stack(outputs, dim=0), None

    def train(self, states, labels):
        """Train the output layer.

        Parameters
        ----------
        states : Tensor
            A batch of hidden states with shape (batch, hidden_size)
        labels : Tensor
            A batch of labels with shape (batch, output_size)
        """
        pinv = torch.pinverse(states.t())
        wout = torch.mm(labels.t(), pinv)
        assert wout.size() == self.out.weight.size()
        self.out.weight = Parameter(wout, requires_grad=False)
