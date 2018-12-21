import numpy as np
import torsk
from torsk.models.numpy_esn import NumpyESN


def model_forward(dtype_str):
    params = torsk.default_params()
    params.dtype = dtype_str
    model = NumpyESN(params)

    inputs = np.ones([3, params.input_size], dtype=dtype_str)
    state = np.zeros([params.hidden_size], dtype=dtype_str)

    outputs, states = model.forward(inputs, state, states_only=False)
    return outputs, states


def test_dtypes():
    for dtype_str in ["float32", "float64"]:
        outputs, states = model_forward(dtype_str)
        dtype = np.dtype(dtype_str)
        assert outputs.dtype == dtype
        assert states.dtype == dtype
