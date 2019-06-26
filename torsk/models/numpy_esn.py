# coding: future_fstrings
from time import time
import logging

from torsk.models.initialize import dense_esn_reservoir
from torsk.models.numpy_map_esn import NumpyMapESNCell, NumpyMapSparseESNCell
import torsk.models.numpy_optimize as opt

logger = logging.getLogger(__name__)

from torsk.numpy_accelerate import *
    
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
        self.wout = bh.zeros(wout_shape, dtype=self.esn_cell.dtype)

        self.ones = bh.ones([1], dtype=self.esn_cell.dtype)
        self.imed_G = None
        self.imed_w = None
        self.imed_V = None

    def forward(self, inputs, state=None, states_only=True):
        if state is None:
            state = bh.zeros([self.esn_cell.hidden_size], dtype=self.esn_cell.dtype)
        if self.params.debug:
            logger.debug("Calling forward function in debug mode")
            return self._forward_debug(inputs, state)
        if states_only:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state)

    def _reset_timers(self):
        self.esn_cell.times = {
            'input_map':0.0,'concatenate':0.0,'state_map':0.0,'tanh':0.0, 'copy':0.0,'dot':0.0
        };
        self.esn_cell.input_map_times = {
            'pixels':0.0,'dct':0.0,'gradient':0.0,'conv':0.0,'random_weights':0.0,'compose':0.0
        };
        
    def _forward_states_only(self, inputs, state):
        print("forward states only")
        t0 = time()
        (T, H) = (len(inputs), len(state))
        
        states = bh.empty((T,H),dtype=bh.float64) # "Bohrium does not support the dtype 'float64'", talk to Mads
        state  = to_bh(state)

        self._reset_timers()
        for i in range(T):
            state = self.esn_cell.forward(inputs[i], state)
            states[i] = state

        t1 = time()
        print(f"finished forward states only: {t1-t0}")
        print(f"times={self.esn_cell.times}\n")
        print(f"input_map_times={self.esn_cell.input_map_times}\n")        
        return None, states
    
    def _forward(self, inputs, state):

        print("forward")
        t0 = time()
        (T,H) = (len(inputs),len(state))
        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty(inputs.shape,dtype=inputs.dtype)

        self._reset_timers()
        for i in range(T):
            inp       = inputs[i]
            state     = self.esn_cell.forward(inp, state)
            t2 = time()
            ext_state = bh.concatenate([self.ones, inp.reshape(-1), state], axis=0) #TODO: What is self.ones doing here??
            t3 = time()
            outputs[i] = bh.dot(self.wout, ext_state).reshape(inputs.shape[1:])
            t4 = time()
            states[i]  = state
            t5 = time()
            self.esn_cell.times['concatenate'] += t3-t2
            self.esn_cell.times['dot'] += t4-t3
            self.esn_cell.times['copy'] += t5-t4
        t1 = time()
        print("finished forward:",t1-t0)
        return outputs, states

    def _forward_debug(self, inputs, state):
        from torsk.visualize import plot_iteration
        print("forward debug")
        t0 = time()

        (T,H) = (len(inputs),len(state))
        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty(inputs.shape,dtype=inputs.dtype)

        print("inputs:",inputs.shape)
        print("state:",state.shape)
        self._reset_timers()
        for i in range(T):
            inp        = inputs[i]
            new_state  = self.esn_cell.forward(inp, state)
            ext_state  = bh.concatenate([self.ones, inp.reshape(-1), new_state], axis=0)
            outputs[i] = bh.dot(self.wout, ext_state).reshape(inputs.shape[1:])
            states[i]  = state

            if (i == 200):
                plot_iteration(self, i, inp, state)

            state = new_state   # TODO: What's going on here?

        t1 = time()
        print("finished forward debug:",t1-t0)
        print(f"times={self.esn_cell.times}\n")        
        return outputs, states

    def predict(self, initial_input, initial_state, nr_predictions):
        print("predict")
        t0 = time()
        (T,H) = (nr_predictions,len(initial_state))
        (M,N) = initial_input.shape
        inp = initial_input
        state = initial_state

        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty((T,M,N),dtype=inp.dtype)

        self._reset_timers()
        for i in range(T):
            state = self.esn_cell.forward(inp, state)
            t2 = time()            
            ext_state = bh.concatenate([self.ones, inp.reshape(-1), state], axis=0)
            t3 = time()
            output = bh.dot(self.wout, ext_state).reshape(M,N)
            t4 = time()
            
            inp = to_np(output)

            outputs[i] = output
            states[i]  = state
            t5 = time()
            
            self.esn_cell.times['concatenate'] += t3-t2
            self.esn_cell.times['dot'] += t4-t3
            self.esn_cell.times['copy'] += t5-t4            
        t1 =time()
        print("finished predicting ",T," states:",t1-t0)
        print(f"times={self.esn_cell.times}\n")
        return outputs, states

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
        
        if self.params.imed_loss:
            from torsk.imed import metric_matrix
            import scipy as sp
            if self.imed_G is None:
                print("Calculating metric matrix...")
                t1 = time()
                self.imed_G = metric_matrix(inputs.shape[1:])
                t2 = time()
                print(f"Computing metric matrix took: {t2-t1}")
                self.imed_w, self.imed_V = sp.linalg.eigh(self.imed_G)
                t3 = time()
                print(f"Diagonalizing metric matrix took: {t3-t2}")
            G, w, V = self.imed_G, self.imed_w, self.imed_V
            S = np.diag(np.sqrt(w))
            G12 = V.dot(S.dot(V.T))

            print("Reprojecting inputs/labels with metric matrix")
            print("G12:",G12.shape)
            print("flat_inputs:",flat_inputs.shape)
            FI = np.matmul(G12, flat_inputs[:,:,None])
            print("FI:",FI.shape)
            FL = np.matmul(G12, flat_labels[:,:,None])            
            print("FL:",FL.shape)
            flat_inputs = FI[:,:,0];
            flat_labels = FL[:,:,0]

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

        if self.params.imed_loss:
            invS12 = bh.diag(1. / bh.sqrt(w))
            invG12 = V.dot(invS12.dot(V.T))
            wout = invG12.dot(wout)

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
        self.dtype = bh.dtype(dtype)

        self.weight_ih = to_bh(np.random.uniform(
            low=-in_weight_init,
            high=in_weight_init,
            size=[hidden_size, input_size])).astype(dtype)

        self.weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(dtype)

        self.bias_ih = to_bh(np.random.uniform(
            low=-in_bias_init,
            high=in_bias_init,
            size=[hidden_size])).astype(dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def forward(self, inputs, state):
        self.check_dtypes(inputs, state)

        # next state
        x_inputs = bh.dot(self.weight_ih, inputs)
        x_state = bh.dot(self.weight_hh, state)
        new_state = bh.tanh(x_inputs + x_state + self.bias_ih)

        return new_state
