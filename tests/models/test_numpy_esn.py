import numpy as np
import torsk
from torsk.models.numpy_esn import NumpyESN, NumpyMapESNCell, NumpyMapSparseESNCell


def model_forward(dtype_str, reservoir):
    params = torsk.default_params()
    params.dtype = dtype_str
    params.reservoir_representation = reservoir
    params.backend = "numpy"
    model = NumpyESN(params)

    inputs = np.ones([3] + params.input_shape, dtype=dtype_str)
    state = np.zeros([model.esn_cell.hidden_size], dtype=dtype_str)

    outputs, states = model.forward(inputs, state, states_only=False)
    return outputs, states


def test_dtypes():
    for dtype_str in ["float32", "float64"]:
        for reservoir in ["dense", "sparse"]:  # TODO: add sparse repr test
            outputs, states = model_forward(dtype_str, reservoir)
            dtype = np.dtype(dtype_str)
            assert outputs.dtype == dtype
            assert states.dtype == dtype


def test_esn_cell():
    input_shape = [10, 12]
    specs = [
        {"type": "pixels", "size": [5, 6], "input_scale": 2},
        {"type": "dct", "size": [5, 5], "input_scale": 2},
        {"type": "conv", "size": [5, 5], "kernel_type": "gauss", "input_scale": 2},
        {"type": "random_weights", "size": [100], "weight_scale": 2}]

    hidden_size = 5 * 6
    hidden_size += 5 * 5
    hidden_size += 6 * 8
    hidden_size += 100
    spectral_radius = 0.5
    density = 1.0
    dtype = "float32"

    dense_cell = NumpyMapESNCell(
        input_shape=input_shape,
        input_map_specs=specs,
        spectral_radius=spectral_radius,
        density=density,
        dtype=dtype)

    assert dense_cell.weight_hh.shape == (hidden_size, hidden_size)

    inputs = np.random.uniform(size=input_shape).astype(dtype)
    state = np.random.uniform(size=[hidden_size]).astype(dtype)

    new_state = dense_cell.forward(inputs, state)
    assert state.shape == new_state.shape
    assert np.any(state != new_state)

    sparse_cell = NumpyMapSparseESNCell(
        input_shape=input_shape,
        input_map_specs=specs,
        spectral_radius=spectral_radius,
        density=density,
        dtype=dtype)

    new_state = sparse_cell.forward(inputs, state)
    assert state.shape == new_state.shape
    assert np.any(state != new_state)


def test_esn():

    # check default parameters
    params = torsk.default_params()

    # check model
    lag_len = 3
    model = NumpyESN(params)
    inputs = np.random.uniform(size=[lag_len] + params.input_shape)
    labels = np.random.uniform(size=[lag_len] + params.input_shape)
    state = np.random.uniform(size=[model.esn_cell.hidden_size])

    # test _forward_states_only
    outputs, states = model.forward(inputs, state, states_only=False)
    assert outputs.shape == (lag_len,) + tuple(params.input_shape)
    assert states.shape == (lag_len, model.esn_cell.hidden_size)

    # test train
    wout = model.wout.copy()
    model.optimize(inputs=inputs, states=states, labels=labels)
    assert not np.any(wout == model.wout)

    # test _forward
    outputs, states = model.predict(inputs[-1], state, nr_predictions=2)
    assert states.shape == (2, model.esn_cell.hidden_size)
    assert outputs.shape == (2,) + tuple(params.input_shape)
