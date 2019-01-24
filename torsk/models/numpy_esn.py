import logging
import numpy as np

from torsk.models.initialize import dense_esn_reservoir
from torsk.models.numpy_map_esn import NumpyMapESNCell, NumpyMapSparseESNCell
import torsk.models.numpy_optimize as opt

logger = logging.getLogger(__name__)


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
            ESNCell = NumpyMapESNCell
        elif params.reservoir_representation == "sparse":
            ESNCell = NumpyMapSparseESNCell

        self.esn_cell = ESNCell(
            input_shape=params.input_shape,
            input_map_specs=params.input_map_specs,
            spectral_radius=params.spectral_radius,
            density=params.density,
            dtype=params.dtype)

        input_size = params.input_shape[0] * params.input_shape[1]
        wout_shape = [input_size, self.esn_cell.hidden_size + input_size + 1]
        self.wout = np.zeros(wout_shape, dtype=self.esn_cell.dtype)

        self.ones = np.ones([1], dtype=self.esn_cell.dtype)

    def forward(self, inputs, state, states_only=True):
        if self.params.debug:
            logger.debug("Calling forward function in debug mode")
            return self._forward_debug(inputs, state)
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
            ext_state = np.concatenate([self.ones, inp.reshape(-1), state], axis=0)
            output = np.dot(self.wout, ext_state)

            outputs.append(output)
            states.append(state)
        return np.asarray(outputs).reshape(inputs.shape), np.asarray(states)

    def _forward_debug(self, inputs, state):
        from torsk.visualize import plot_iteration
        outputs, states = [], []
        idx = 0
        for inp in inputs:
            new_state = self.esn_cell.forward(inp, state)
            ext_state = np.concatenate([self.ones, inp.reshape(-1), new_state], axis=0)
            output = np.dot(self.wout, ext_state)

            outputs.append(output)
            states.append(new_state)

            if (idx == 200):
                new_state = self.esn_cell.forward(inp, state)
                input_stack = self.esn_cell.input_map(inp)
                x_input = self.esn_cell.cat_input_map(input_stack)
                x_state = self.esn_cell.state_map(state)

                plot_iteration(self, idx, inp, state, new_state, input_stack, x_input, x_state)

            state = new_state
            idx += 1
        return np.asarray(outputs).reshape(inputs.shape), np.asarray(states)

    def predict(self, initial_inputs, initial_state, nr_predictions):
        inp_shape = initial_inputs.shape
        inp = initial_inputs
        state = initial_state
        outputs, states = [], []

        for ii in range(nr_predictions):
            state = self.esn_cell.forward(inp, state)
            ext_state = np.concatenate([self.ones, inp.reshape(-1), state], axis=0)
            output = np.dot(self.wout, ext_state).reshape(inp_shape)

            inp = output

            outputs.append(output)
            states.append(state)
        return np.asarray(outputs), np.asarray(states)

    def optimize(self, inputs, states, labels):
        """Train the output layer.

        Parameters
        ----------
        inputs : Tensor
            A batch of inputs with shape (batch, ydim, xdim)
        states : Tensor
            A batch of hidden states with shape (batch, hidden_size)
        labels : Tensor
            A batch of labels with shape (batch, ydim, xdim)
        """
        method = self.params.train_method
        beta = self.params.tikhonov_beta

        train_length = inputs.shape[0]
        flat_inputs = inputs.reshape([train_length, -1])
        flat_labels = labels.reshape([train_length, -1])

        if True:
            from torsk.imed import metric_matrix
            logger.debug("Calculating metric matrix")
            G = metric_matrix(inputs.shape[1:])
            w, V = np.linalg.eigh(G)
            W = np.diag(w**.5)
            G12 = V.dot(W.dot(V.T))

            logger.debug("Reprojecting inputs/labels with metric matrix")
            flat_inputs = np.matmul(G12, flat_inputs[:,:,None])[:,:,0]
            flat_labels = np.matmul(G12, flat_labels[:,:,None])[:,:,0]

        if method == 'tikhonov':
            if beta is None:
                raise ValueError(
                    'For Tikhonov training the beta parameter cannot be None.')
            logger.debug(f"Tikhonov optimizing with beta={beta}")
            wout = opt.tikhonov(flat_inputs, states, flat_labels, beta)

        elif 'pinv' in method:
            if beta is not None:
                logger.debug("With pseudo inverse training the "
                             "beta parameter has no effect.")
            logger.debug(f"Pinv optimizing with mode={method}")
            wout = opt.pseudo_inverse(
                flat_inputs, states, flat_labels,
                mode=method.replace("pinv_", ""))

        else:
            raise ValueError(f'Unkown training method: {method}')

        if(wout.shape != self.wout.shape):
            raise ValueError("Optimized and original Wout shape do not match."
                             f"{wout.shape} / {self.wout.shape}")
        self.wout = wout


class NumpyStandardESNCell(object):
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
    def __init__(self, input_size, hidden_size, spectral_radius,
                 in_weight_init, in_bias_init, density, dtype):

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
            size=[hidden_size]).astype(dtype)

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
