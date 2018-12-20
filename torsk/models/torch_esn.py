import pathlib
import logging

import numpy as np
from scipy.linalg import lstsq, svd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.rnn import RNNCellBase

from torsk import Params
from torsk.models.initialize import (
        dense_esn_reservoir, scale_weight, sparse_esn_reservoir)

_module_dir = pathlib.Path(__file__).absolute().parent
logger = logging.getLogger(__name__)


def get_default_params():
    return Params(_module_dir / 'default_esn_params.json')


def _extended_states(inputs, states):
    ones = torch.ones([inputs.size(0), 1])
    return torch.cat([ones, inputs, states], dim=1).t()


def pseudo_inverse_lstsq(inputs, states, labels):
    X = _extended_states(inputs, states)

    wout, _, _, s = lstsq(X.t(),labels)
    condition = s[0]/s[-1]

    if(np.log2(np.abs(condition)) > 12):  # More than half of the bits in the data are lost
        logger.warning(
            f"Large condition number in pseudoinverse: {condition}"
            " losing more than half of the digits. Expect numerical blowup!")
        logger.warning(f"Largest and smallest singular values: {s[0]}  {s[-1]}")
        
    return torch.Tensor(wout.T)

def pseudo_inverse_svd(inputs, states, labels):
    X = _extended_states(inputs, states)

    U, s, Vh = svd(X.numpy())
    L = labels.numpy().T
    condition = s[0] / s[-1]

    scale = s[0]
    n = len(s[np.abs(s / scale) > 1e-4])  # Ensure condition number less than 10.000
    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = np.dot(np.dot(L, v) * (1 / s[:n]), uh)
    return torch.Tensor(wout)
    

def pseudo_inverse(inputs, states, labels):
    return pseudo_inverse_svd(inputs, states, labels)


def tikhonov(inputs, states, labels, beta):
    X = _extended_states(inputs, states)

    Id  = torch.eye(X.size(0))
    A = torch.mm(X, X.t()) + beta * Id
    B = torch.mm(X, labels)
    # Solve linear system instead of calculating inverse
    wout,_ = torch.gesv(B, A)
    return wout.t()


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

    def forward(self, inputs, state):
        self.check_forward_input(inputs)
        self.check_forward_hidden(inputs, state)
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
            raise ValueError("SparseTorchESNCell can only process batch_size==1")

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
            esn_cell = TorchESNCell
        elif params.reservoir_representation == "sparse":
            esn_cell = SparseTorchESNCell

        self.esn_cell = esn_cell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init,
            in_bias_init=params.in_bias_init,
            density=params.density,
            dtype=params.dtype)

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
            wout = tikhonov(inputs, states, labels, beta)

        elif method == 'pinv':
            if beta is not None:
                logger.debug('With pseudo inverse training the '
                             'beta parameter has no effect.')
            wout = pseudo_inverse(inputs, states, labels)

        else:
            raise ValueError(f'Unkown training method: {method}')

        if(wout.size() != self.out.weight.size()):
            print("wout:",wout.shape,", weight:",self.out.weight.shape);
            raise;
        
        self.out.weight = Parameter(wout, requires_grad=False)


def train_predict_esn(model, dataset, outdir=None, shuffle=True):

    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = model.params.transient_length
    model.eval()  # because we are not using gradients

    ii = np.random.randint(low=0, high=len(dataset)) if shuffle else 0
    inputs, labels, pred_labels, orig_data = dataset[ii]

    inputs = inputs.unsqueeze(1)
    labels = labels.unsqueeze(1)
    pred_labels = pred_labels.unsqueeze(1)

    logger.debug(f"Creating {inputs.size(0)} training states")
    zero_state = torch.zeros([1, model.esn_cell.hidden_size], dtype=model.esn_cell.dtype)
    _, states = model.forward(inputs, zero_state, states_only=True)

    logger.debug("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    logger.debug(f"Predicting the next {model.params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, out_states = model.predict(
        init_inputs, states[-1], nr_predictions=model.params.pred_length)

    # if outdir is not None:
    #     outfile = outdir / "train_data.nc"
    #     logger.debug(f"Saving training to {outfile}")
    #     dump_training(
    #         outfile,
    #         inputs=inputs.reshape([-1, params.input_size]),
    #         labels=labels.reshape([-1, params.output_size]),
    #         states=states.reshape([-1, params.hidden_size]),
    #         pred_labels=pred_labels.reshape([-1, params.output_size]))

    # logger.debug("Optimizing output weights")
    # model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    # if outdir is not None:
    #     modelfile = outdir / "model.pth"
    #     logger.debug(f"Saving model to {modelfile}")
    #     save_model(modelfile.parent, model)

    # logger.debug(f"Predicting the next {params.pred_length} frames")
    # init_inputs = labels[-1]
    # outputs, out_states = model.predict(
    #     init_inputs, states[-1], nr_predictions=model.params.pred_length)

    # if outdir is not None:
    #     outfile = outdir / "pred_data.nc"
    #     logger.debug(f"Saving prediction to {outfile}")
    #     dump_prediction(outfile,
    #         outputs=outputs.reshape([-1, params.input_size]),
    #         labels=pred_labels.reshape([-1, params.output_size]),
    #         states=out_states.reshape([-1, params.hidden_size]))

    return model, outputs, pred_labels

