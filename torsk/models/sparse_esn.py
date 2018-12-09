import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase

from torsk.models.utils import scale_weight, sparse_esn_reservoir


class SparseESNCell(RNNCellBase):
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
        super(SparseESNCell, self).__init__(
            input_size, hidden_size, bias=True, num_chunks=1)

        self.input_size = input_size
        self.hidden_size = hidden_size

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

    def forward(self, inputs, state):
        if not inputs.size(0) == state.size(0) == 1:
            raise ValueError("SparseESNCell can only process batch_size==1")

        # reshape for matrix multiplication
        inputs = inputs.reshape([-1, 1])
        state = state.reshape([-1, 1])

        # next state
        x_inputs = torch.mm(self.weight_ih, inputs)
        x_state = torch.mm(self.weight_hh, state)
        new_state = torch.tanh(x_inputs + x_state + self.bias_ih)

        return new_state.reshape([1, -1])
