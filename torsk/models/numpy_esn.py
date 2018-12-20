import pathlib
import logging

import numpy as np

from torsk import Params
from torsk.models.initialize import dense_esn_reservoir, sparse_esn_reservoir
from torsk.models.optimize import pseudo_inverse

_module_dir = pathlib.Path(__file__).absolute().parent
logger = logging.getLogger(__name__)


class NumpyESNCell(object):
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
    input : array
        contains input features of shape (batch, input_size)
    state : array
        current hidden state of shape (batch, hidden_size)

    Outputs
    -------
    state' : array
        contains the next hidden state of shape (batch, hidden_size)
    """
    def __init__(
            self, input_size, hidden_size,
            spectral_radius, in_weight_init, in_bias_init, density, dtype):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = np.dtype(dtype)

        self.weight_ih = np.random.uniform(
            low=-in_weight_init,
            high=in_weight_init,
            size=[hidden_size, input_size]).astype(dtype)

        self.weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(dtype)

        self.bias_ih = np.random.uniform(
            low=-in_bias_init,
            high=in_bias_init,
            size=[hidden_size,]).astype(dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def forward(self, inputs, state):
        self.check_dtypes(inputs, state)

        # next state
        x_inputs = np.dot(self.weight_ih, inputs)
        x_state = np.dot(self.weight_hh, state)
        new_state = np.tanh(x_inputs + x_state + self.bias_ih)

        return new_state


class NumpyESN(object):
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
        super(NumpyESN, self).__init__()
        self.params = params

        if params.reservoir_representation == "dense":
            esn_cell = NumpyESNCell

        self.esn_cell = esn_cell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init,
            in_bias_init=params.in_bias_init,
            density=params.density,
            dtype=params.dtype)

        self.wout = np.zeros(
            [params.input_size, params.hidden_size + params.input_size + 1])

        self.ones = np.ones([1,])
        
    def forward(self, inputs, state, states_only=True):
        if states_only:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state)

    def _forward_states_only(self, inputs, state):
        states = []
        for inp in inputs:
            state = self.esn_cell.forward(inp, state)
            states.append(state)
        return None, np.asarray(states)

    def _forward(self, inputs, state):
        outputs, states = [], []
        for inp in inputs:
            state = self.esn_cell.forward(inp, state)
            ext_state = np.concatenate([self.ones, inp, state], axis=0)
            output = np.dot(self.wout, ext_state)
            outputs.append(output)
            states.append(state)
        return np.asarray(outputs), np.asarray(states)

    def predict(self, initial_inputs, initial_state, nr_predictions):
        inp = initial_inputs
        state = initial_state
        outputs, states = [], []

        for ii in range(nr_predictions):
            state = self.esn_cell.forward(inp, state)
            ext_state = np.concatenate([self.ones, inp, state], axis=0)
            output = np.dot(self.wout, ext_state)
            inp = output

            outputs.append(output)
            states.append(state)
        return np.asarray(outputs), np.asarray(states)

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
        method = self.params.train_method
        beta = self.params.tikhonov_beta

        if method == 'tikhonov':
            if beta is None:
                raise ValueError(
                    'For Tikhonov training the beta parameter cannot be None.')
            wout = tikhonov(inputs, states, labels, beta)

        elif method == 'pinv':
            if beta is not None:
                logger.debug('With pseudo inverse training the '
                             'beta parameter has no effect.')
            wout = pseudo_inverse(inputs, states, labels)

        else:
            raise ValueError(f'Unkown training method: {method}')

        if(wout.shape != self.wout.shape):
            print("wout:",wout.shape,", weight:",self.out.weight.shape);
            raise;
        
        self.wout = wout
