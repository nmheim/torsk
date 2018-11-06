import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase


def connection_mask(dim, density, symmetric):
    mask = np.random.uniform(low=0., high=1., size=(dim, dim)) < density
    if symmetric:
        triu = np.triu(mask, k=1)
        tril = np.tril(mask.T)
        mask = triu + tril
    return mask


def dense_esn_reservoir(dim, spectral_radius, density, symmetric):
    """Creates a dense square matrix with random non-zero elements according
    to the density parameter and a given spectral radius.
    Params:
        dim(int): int that specifies the dimensions of the square matrix
        spectral_radius(float): spectral radius of the created matrix
        symmetric(bool): defines if the created matrix is symmetrix or not
    Returns:
        square ndarray
    """
    mask = connection_mask(dim, density, symmetric)
    res  = np.random.uniform(low=-1., high=1., size=(dim, dim))
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
            self, input_size, hidden_size, spectral_radius, in_weight_init):
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
            density=1.0, symmetric=False)
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



