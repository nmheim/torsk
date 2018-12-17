import numpy as orig_numpy
import bohrium as np
#import numpy as np
import scipy as sp
from scipy import sparse
from scipy.stats import uniform
import json
import pathlib,logging
import matplotlib.pyplot as plt
import sys;

DTYPE=np.float32
logger = logging.getLogger(__name__)

from scipy.fftpack import dct, idct


# Least-squares approximation to restricted DCT-III / Inverse DCT-II
def sct_basis(nx,nk):
    xs = np.arange(nx);
    ks = np.arange(nk);
    basis = 2*np.cos(np.pi*(xs[:,None]+0.5)*ks[None,:]/nx);        
    return basis;

def sct(fx,basis):  
    fk,_,_,_ = lstsq(basis,fx);
    return fk;

def isct(fk,basis):
    fx = np.dot(basis,fk);
    return fx;

def sct2(Fxx,basis1, basis2):
    Fkx = sct(Fxx.T,basis2);
    Fkk = sct(Fkx.T,basis1);
    return Fkk

def isct2(Fkk,basis1, basis2):
    Fkx = isct(Fkk.T,basis2);
    Fxx = isct(Fkx.T,basis1);
    return Fxx

def dct2(Fxx,nk1,nk2):
    Fkk = dctn(Fxx,norm='ortho')[:nk1,:nk2];
    return Fkk

def idct2(Fkk,nx1,nx2):
    Fxx = idctn(Fkk,norm='ortho',shape=(nx1,nx2));
    return Fxx

def idct2_sequence(Ftkk,xsize):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftkk : ndarray with shape (time, nk1, nk2)
    size : (ny,nx) determines the resolution of the image

    Returns
    -------
    Ftxx: ndarray with shape (time, ny,nx)
    """        
    Ftxx = idctn(Ftkk,norm='ortho',shape=xsize,axes=[1,2]);
    return Ftxx;


def dct2_sequence(Ftxx, ksize):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftxx : ndarray with shape (time, ydim, xdim)
    size : (nk1,nk2) determines how many DCT coefficents are kept

    Returns
    -------
    Ftkk: ndarray with shape (time, nk1, nk2)
    """    
    Ftkk = dctn(Ftxx,norm='ortho',axes=[1,2])[:,:ksize[0],:ksize[1]];
    return Ftkk;


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
    # res = np.random.uniform(low=-1.0, high=1.0, size=[dim, dim])
    if symmetric:
        res = np.triu(res) + np.tril(res.T, k=-1)
    res *= mask.astype(float)
    if spectral_radius:
        eig = orig_numpy.linalg.eigvals(res.copy2numpy())
        rho = np.abs(eig).max()
        res = spectral_radius * res / rho
    return res


def scale_weight(weight, value):
    """Scales the weight matrix to (-value, value)"""
    weight *= 2 * value
    weight -= value
    return weight


def sparse_esn_reservoir(dim, spectral_radius, density, symmetric):
    """Creates a CSR representation of a sparse ESN reservoir.
    Params:
        dim: int, dimension of the square reservoir matrix
        spectral_radius: float, largest eigenvalue of the reservoir matrix
        density: float, 0.1 corresponds to approx every tenth element
            being non-zero
        symmetric: specifies if matrix.T == matrix
    Returns:
        matrix: a square scipy.sparse.csr_matrix
    """
    rvs = uniform(loc=-1., scale=2.).rvs
    matrix = sparse.random(dim, dim, density=density, data_rvs=rvs)
    matrix = matrix.tocsr()
    if symmetric:
        matrix = sparse.triu(matrix)
        tril = sparse.tril(matrix.transpose(), k=-1)
        matrix = matrix + tril
        # calc eigenvalues with scipy's lanczos implementation:
        eig, _ = sparse.linalg.eigsh(matrix, k=2, tol=1e-4)
    else:
        eig, _ = sparse.linalg.eigs(matrix, k=2, tol=1e-4)

    rho = np.abs(eig).max()
    matrix = matrix.multiply(1. / rho)
    matrix = matrix.multiply(spectral_radius)
    return matrix

#from bohrium import lapack

def tikhonov(inputs, states, labels, beta):
    X = _extended_states(inputs, states)

    Id  = np.eye(X.shape[0])
    A = np.dot(X, X.T) + beta * Id
    B = np.dot(X, labels)
    # Solve linear system instead of calculating inverse
#    wout = lapack.gesv(A.copy(), B.copy())
    wout = sp.linalg.gesv(A.copy(), B.copy())
    return wout.T


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
    print("_extended_states: ones",ones.shape,"inputs",inputs.shape,"states",states.shape)
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
            size=[hidden_size, input_size]).astype(DTYPE)

        self.weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(DTYPE)

        self.bias_ih = np.random.uniform(
            low=-in_bias_init,
            high=in_bias_init,
            size=[hidden_size, 1]).astype(DTYPE)

        print("Dense:",self.weight_hh.shape)

    def forward(self, inputs, state):
        # reshape for matrix multiplication
        inputs = inputs.reshape([-1, 1]).astype(DTYPE)
        state = state.reshape([-1, 1]).astype(DTYPE)

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
        if params.reservoir_representation == "sparse":
            esn_cell = SparseESNCell            

        self.esn_cell = esn_cell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init,
            in_bias_init=params.in_bias_init,
            density=params.density)

        self.wout = np.zeros(
            [params.output_size, params.hidden_size + params.input_size + 1])

        self.ones = np.ones([1,])
        
        
    def forward(self, inputs, state, states_only=True):

        if states_only:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state)

    def _forward_states_only(self, inputs, state):
        states = []
#        print("inputs.shape:",inputs.shape);
        for inp in inputs:
            state = self.esn_cell.forward(inp, state)
            states.append(state)
        return None, np.asarray(states).squeeze()

    def _forward(self, inputs, state):
        outputs, states = [], []
        for inp in inputs:
            state = self.esn_cell.forward(inp, state).squeeze()
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
            state = self.esn_cell.forward(inp, state).squeeze()
#            print("predict: ones",self.ones.shape,"inp",inp.shape,"state",state.shape,"wout",self.wout.shape)
            ext_state = np.concatenate([self.ones, inp, state], axis=0)
            output = np.dot(self.wout,ext_state)
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
        # if len(inputs.shape) == 3:
        #     inputs = inputs.reshape([-1, inputs.shape[2]])
        #     states = states.reshape([-1, states.shape[2]])
        #     labels = labels.reshape([-1, labels.shape[2]])

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

    inputs, labels, pred_labels, orig_data = loader[0]
 
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

def plot_mackey(predictions, labels, weights=None):

    def sort_output(output, error):
        sort = sorted(zip(output, error), key=lambda arg: arg[1].sum())
        sort = np.array(sort)
        sort_out, sort_err = sort[:, 0, :], sort[:, 1, :]
        return sort_out, sort_err

    error = np.abs(predictions - labels)
    predictions, _ = sort_output(predictions, error)
    labels, error = sort_output(labels, error)

    if weights is None:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(4, 1)
        hist, bins = np.histogram(weights, bins=100)
        ax[3].plot(bins[:-1], hist, label=r"$W^{out}$ histogram")

    ax[0].plot(labels[0], label="Truth")
    ax[0].plot(predictions[0], label="Prediction")
    ax[1].plot(labels[-1])
    ax[1].plot(predictions[-1])

    mean, std = error.mean(axis=0), error.std(axis=0)
    ax[2].plot(mean, label=r"Mean Error $\mu$")
    ax[2].fill_between(
        np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.5,
        label=r"$\mu \pm \sigma$")

    ax[0].set_ylim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    ax[2].set_ylim(-0.1, 1.1)

    for a in ax:
        a.legend()

    plt.tight_layout()
    return fig, ax




class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, params):
        """Updates parameters based on a dictionary."""
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def _simulate_mackey(b=None, N=3000):
    c = 0.2
    tau = 17
    n = 10

    yinit = np.array([0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076,
                      1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
                      1.0739, 1.0759])

    if b is None:
        b = np.zeros(N) + 0.1

    y = np.zeros(N)
    y[:yinit.shape[0]] = yinit

    for i in range(tau, N - 1):
        yi = y[i] - b[i] * y[i] + c * y[i - tau] / (1 + y[i - tau]**n)
        y[i + 1] = yi
    return y

class MackeyDataset:
    def __init__(self, train_length, pred_length, simulation_steps):
        if simulation_steps <= train_length + pred_length:
            raise ValueError('simulation_steps must be larger than seq_length.')

        self.simulation_steps = simulation_steps
        self.train_length = train_length
        self.pred_length = pred_length
        self.nr_sequences = self.simulation_steps \
            - self.train_length - self.pred_length

        self.seq = normalize(_simulate_mackey(N=simulation_steps))
        self.seq = self.seq.reshape((-1, 1))

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('MackeyDataset index out of range.')
        sub_seq = self.seq[index:index + self.train_length + self.pred_length + 1]
        inputs, labels, pred_labels = split_train_label_pred(
            sub_seq, self.train_length, self.pred_length)
        return inputs, labels, pred_labels, None

    def __len__(self):
        return self.nr_sequences

def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels
    

    
def normalize(data, vmin=None, vmax=None):
    """Normalizes data to values from 0 to 1.
    If vmin/vmax are given they are assumed to be the maximal
    values of data"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return (data - vmin) / np.abs(vmin - vmax)


def gauss2d(center, sigma, size, borders=[[-2, 2], [-2, 2]]):
    yc, xc = center
    yy = np.linspace(borders[0][0], borders[0][1], size[0])-yc
    xx = np.linspace(borders[1][0], borders[1][1], size[1])-xc
#    yy, xx = np.meshgrid(yy, xx)
#    gauss = ((xx - xc)**2 + (yy - yc)**2) / (2 * sigma)**2

    gauss = (xx[:,None]**2 + yy[None,:]**2) / (2*sigma**2)

    return np.exp(-gauss)

def gauss2D(centers, sigma, size, borders=[[-2, 2], [-2, 2]]):
    yc, xc = centers[:,0], centers[:,1]
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])

    gauss = ((xx[None,:,None]-xc[:,None,None])**2 + (yy[None,None,:]-yc[:,None,None])**2) / (2*sigma**2)

    return np.exp(-gauss)

class CircleDataset:
    def __init__(self, params, center, sigma):

        self.train_length = params.train_length
        self.pred_length = params.pred_length
        self.domain = params.domain
        self.nr_sequences = center.shape[0] - self.train_length - self.pred_length

        xsize = params.xsize
        ksize = params.ksize

        print(f"gauss2d({center},{sigma},{xsize})")
        #seq = np.array([gauss2d(c, sigma, xsize) for c in center])
        seq = gauss2D(center,sigma,xsize);
        print("normalize")
        seq = normalize(seq)

        if self.domain == "DCT":
            if ksize is None:
                ksize = xsize
            seq = dct2_sequence(seq, ksize)
            seq = seq.reshape((-1, ksize[0] * ksize[1]))
        else:
            seq = seq.reshape((-1, xsize[0] * xsize[1]))

        self.seq = seq
        self.xsize = xsize
        self.ksize = ksize

    def to_image(self, seq):
        if(self.domain == "DCT"):
            seq = seq.reshape((-1, self.ksize[0], self.ksize[1]))
            return idct2_sequence(seq, self.xsize)
        else:
            return seq.reshape((-1, self.xsize[0], self.xsize[1]))

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('CircleDataset index out of range.')
        sub_seq = self.seq[index:index + self.train_length + self.pred_length + 1]
        inputs, labels, pred_labels = split_train_label_pred(
            sub_seq, self.train_length, self.pred_length)
        return inputs, labels, pred_labels, None

    def __len__(self):
        return self.nr_sequences



class SparseESNCell(object):
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

        self.input_size = input_size
        self.hidden_size = hidden_size

        # input matrix
        in_weight = np.random.uniform(size=[hidden_size, input_size],low=-in_weight_init,high=in_weight_init)
        self.weight_ih = in_weight

        # sparse reservoir matrix
        matrix = sparse_esn_reservoir(
            dim=hidden_size,
            spectral_radius=spectral_radius,
            density=density,
            symmetric=False)

        self.weight_hh = matrix.tocsr()
        # biases
        self.bias_ih = np.random.uniform(size=[hidden_size, 1],low=-in_bias_init,high=in_bias_init)

        print("Sparse:",self.weight_hh.shape)

    def forward(self, inputs, state):
        # reshape for matrix multiplication
        inputs = inputs.reshape([-1, 1]).astype(DTYPE)
        state = state.reshape([-1, 1]).astype(DTYPE)

        # next state
        x_inputs = np.dot(self.weight_ih,inputs)
        x_state  = self.weight_hh * state
        new_state = np.tanh(x_inputs + x_state + self.bias_ih)

        return new_state.reshape([1, -1])
        
    
