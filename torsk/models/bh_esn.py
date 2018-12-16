import pathlib
import logging

import numpy as np
import scipy as sp

from torsk import Params
from torsk.models.utils import (
        dense_esn_reservoir, scale_weight, sparse_esn_reservoir)

_module_dir = pathlib.Path(__file__).absolute().parent
logger = logging.getLogger(__name__)


def pseudo_inverse_svd(inputs, states, labels):
    X = _extended_states(inputs, states)

    U, s, Vh = sp.linalg.svd(X)
    L = labels.T
    condition = s[0] / s[-1]

    scale = s[0]
    n = len(s[np.abs(s / scale) > 1e-4])  # Ensure condition number less than 10.000
    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = np.dot(np.dot(L, v) * (1 / s[:n]), uh)
    return wout
    

def pseudo_inverse(inputs, states, labels):
    return pseudo_inverse_svd(inputs, states, labels)


def _extended_states(inputs, states):
    ones = np.ones([inputs.shape[0], 1])
    return np.concatenate([ones, inputs, states], axis=1).T


class ESNCell(object):
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
            spectral_radius, in_weight_init, in_bias_init, density):

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = np.random.uniform(
            low=-in_weight_init,
            high=in_weight_init,
            size=[hidden_size, input_size])

        self.weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)

        self.bias_ih = np.random.uniform(
            low=-in_bias_init,
            high=in_bias_init,
            size=[hidden_size, 1])

    def forward(self, inputs, state):
        # reshape for matrix multiplication
        inputs = inputs.reshape([-1, 1])
        state = state.reshape([-1, 1])

        # next state
        x_inputs = np.dot(self.weight_ih, inputs)
        x_state = np.dot(self.weight_hh, state)
        new_state = np.tanh(x_inputs + x_state + self.bias_ih)

        return new_state.reshape([1, -1])


class ESN(object):
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
        self.params = params
        if params.input_size != params.output_size:
            raise ValueError(
                "Currently input and output dimensions must be the same.")

        if params.reservoir_representation == "dense":
            esn_cell = ESNCell

        self.esn_cell = esn_cell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init,
            in_bias_init=params.in_bias_init,
            density=params.density)

        self.wout = np.zeros(
            [params.output_size, params.hidden_size + params.input_size + 1])

        self.ones = np.ones([1, 1])
        
        
    def forward(self, inputs, state, states_only=True):
        if inputs.shape[1] != 1:
            raise ValueError("Supports only batch size of one -.-")
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
            ext_state = np.concatenate([self.ones, inp, state], axis=1).T
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
            ext_state = np.concatenate([self.ones, inp, state], axis=1).T
            print(ext_state.dtype)
            raise
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
            A batch of labels with shape (batch, output_size)
        """
        if len(inputs.shape) == 3:
            inputs = inputs.reshape([-1, inputs.shape[2]])
            states = states.reshape([-1, states.shape[2]])
            labels = labels.reshape([-1, labels.shape[2]])

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


def train_predict_esn(model, loader, params, outdir=None):
    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = params.transient_length

    inputs, labels, pred_labels, orig_data = next(loader)
    inputs = inputs.numpy()
    labels = labels.numpy()
    pred_labels = pred_labels.numpy()

    logger.debug(f"Creating {inputs.shape[0]} training states")
    zero_state = np.zeros([model.esn_cell.hidden_size])
    _, states = model.forward(inputs, zero_state, states_only=True)

    logger.debug("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    logger.debug(f"Predicting the next {params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, out_states = model.predict(
        init_inputs, states[-1], nr_predictions=params.pred_length)

    return model, outputs, pred_labels, orig_data
