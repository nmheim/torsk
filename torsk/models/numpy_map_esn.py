import logging
import numpy as np
from scipy.signal import convolve2d

from torsk.data.conv import get_kernel, conv2d_output_shape
from torsk.data.utils import resample2d
from torsk.data.transform import dct2
from torsk.models.initialize import dense_esn_reservoir, sparse_nzpr_esn_reservoir
from torsk.sparse import SparseMatrix

logger = logging.getLogger(__name__)


def _input_map(image, input_map_specs):
    features = []
    for spec in input_map_specs:
        if spec["type"] == "pixels":
            _features = resample2d(image, spec["size"]).reshape(-1)
            _features = spec["input_scale"] * _features
        elif spec["type"] == "dct":
            _features = dct2(image, *spec["size"]).reshape(-1)
            _features = spec["input_scale"] * _features
        elif spec["type"] == "conv":
            _features = convolve2d(
                image, spec["kernel"], mode="valid").reshape(-1)
            _features = spec["input_scale"] * _features
        elif spec["type"] == "random_weights":
            _features = np.dot(spec["weight_ih"], image.reshape(-1))
        else:
            raise ValueError(spec)
        features.append(_features)
    return np.concatenate(features, axis=0)


def _get_hidden_size(input_shape, input_map_specs):
    hidden_size = 0
    for spec in input_map_specs:
        if spec["type"] == "conv":
            _shape = conv2d_output_shape(input_shape, spec["size"])
            hidden_size += _shape[0] * _shape[1]
        elif spec["type"] == "random_weights":
            hidden_size = spec["size"]
        else:
            shape = spec["size"]
            hidden_size += shape[0] * shape[1]
    return hidden_size


def _init_input_map_specs(input_map_specs, input_shape):
    for spec in input_map_specs:
        if spec["type"] == "conv":
            spec["kernel"] = get_kernel(spec["size"], spec["kernel_type"])
        elif spec["type"] == "random_weights":
            spec["weight_ih"] = np.random.uniform(
                low=-spec["weight_scale"],
                high=spec["weight_scale"],
                size=[spec["size"], input_shape[0] * input_shape[1]])
    return input_map_specs


class NumpyMapESNCell(object):
    """An Echo State Network (ESN) cell that enables custom input -> state
    mappings.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images
    input_map_specs : list
        a list of dicts that specify the input mapping. example:
        [{"type": "random_weights", "size": 500, "weight_scale": 1.0},
         {"type": "pixels", "size": [10, 10]},
         {"type": "dct", "size": [10, 10]},
         {"type": "conv", "size": [2, 2], "kernel_type": "gauss"}]
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

    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype):
        self.input_shape = input_shape
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = np.dtype(dtype)

        self.hidden_size = self.get_hidden_size(input_shape)
        logger.info(f"ESN hidden size: {self.hidden_size}")
        self.weight_hh = dense_esn_reservoir(
            dim=self.hidden_size, spectral_radius=self.spectral_radius,
            density=self.density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(self.dtype)

        self.input_map_specs = _init_input_map_specs(input_map_specs)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return _get_hidden_size(input_shape, input_map_specs)

    def input_map(self, image):
        return _input_map(image, self.input_map_specs)

    def forward(self, image, state):
        self.check_dtypes(image, state)

        x_input = self.input_map(image)
        x_state = np.dot(self.weight_hh, state)
        new_state = np.tanh(x_input + x_state)

        return new_state


class NumpyMapSparseESNCell(object):
    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype):
        self.input_shape = input_shape
        self.input_map_specs = input_map_specs
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = np.dtype(dtype)

        self.hidden_size = self.get_hidden_size(input_shape)
        logger.info(f"ESN hidden size: {self.hidden_size}")
        nonzeros_per_row = int(self.hidden_size * density)
        self.weight_hh = sparse_nzpr_esn_reservoir(
            dim=self.hidden_size,
            spectral_radius=self.spectral_radius,
            nonzeros_per_row=nonzeros_per_row,
            dtype=self.dtype)

        for spec in self.input_map_specs:
            if spec["type"] == "conv":
                spec["kernel"] = get_kernel(spec["size"], spec["kernel_type"])
            elif spec["type"] == "random_weights":
                spec["weight_ih"] = np.random.uniform(
                    low=-spec["weight_scale"],
                    high=spec["weight_scale"],
                    size=[spec["size"], input_shape[0] * input_shape[1]])

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return _get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        return _input_map(image, self.input_map_specs)

    def forward(self, image, state):
        self.check_dtypes(image, state)

        x_input = self.input_map(image)
        x_state = self.weight_hh.sparse_dense_mv(state)
        new_state = np.tanh(x_input + x_state)

        return new_state



