import numpy as np
import torsk
from torsk.models.numpy_esn import NumpyESN, NumpyESNCell


def model_forward(dtype_str, reservoir):
    params = torsk.default_params()
    params.dtype = dtype_str
    params.reservoir_representation = reservoir
    params.hidden_size = 100
    params.input_size = 10
    params.backend = "numpy"
    model = NumpyESN(params)

    inputs = np.ones([3, params.input_size], dtype=dtype_str)
    state = np.zeros([params.hidden_size], dtype=dtype_str)

    outputs, states = model.forward(inputs, state, states_only=False)
    return outputs, states


def test_dtypes():
    for dtype_str in ["float32", "float64"]:
        for reservoir in ["dense"]:  # TODO: add sparse repr test
            outputs, states = model_forward(dtype_str, reservoir)
            dtype = np.dtype(dtype_str)
            assert outputs.dtype == dtype
            assert states.dtype == dtype


def test_esn_cell():
    input_size = 1
    hidden_size = 10
    spectral_radius = 0.5
    weight_init = 0.1
    bias_init = 0.1
    density = 1.0

    dense_cell = NumpyESNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        spectral_radius=spectral_radius,
        in_weight_init=weight_init,
        in_bias_init=bias_init,
        density=density,
        dtype="float64")

    assert dense_cell.weight_hh.shape == (hidden_size, hidden_size)
    assert dense_cell.weight_ih.shape == (hidden_size, input_size)

    inputs = np.random.uniform(size=[input_size])
    state = np.random.uniform(size=[hidden_size])

    new_state = dense_cell.forward(inputs, state)
    assert state.shape == new_state.shape
    assert np.any(state != new_state)

    # TODO:  test sparse esn_cell once it exists
    # sparse_cell = NumpySparseESNCell(
    #     input_size=input_size,
    #     hidden_size=hidden_size,
    #     spectral_radius=spectral_radius,
    #     in_weight_init=weight_init,
    #     in_bias_init=bias_init,
    #     density=density,
    #     dtype="float64")

    # inputs = torch.rand([1, input_size])
    # state = torch.rand(1, hidden_size)

    # new_state = sparse_cell(inputs, state)
    # assert state.shape == new_state.shape
    # assert np.any(state != new_state)


def test_esn():

    # check default parameters
    params = torsk.default_params()
    params.input_size = 10
    params.hidden_size = 100

    # check model
    lag_len = 3
    model = NumpyESN(params)
    inputs = np.random.uniform(size=[lag_len, params.input_size])
    labels = np.random.uniform(size=[lag_len, params.input_size])
    state = np.random.uniform(size=[params.hidden_size])

    # test _forward_states_only
    outputs, states = model.forward(inputs, state, states_only=False)
    assert outputs.shape == (lag_len, params.input_size)
    assert states.shape == (lag_len, params.hidden_size)

    # test train
    wout = model.wout.copy()
    model.optimize(inputs=inputs, states=states, labels=labels)
    assert not np.any(wout == model.wout)

    # test _forward
    outputs, states = model.predict(inputs[-1], state, nr_predictions=2)
    assert states.shape == (2, params.hidden_size)
    assert outputs.shape == (2, params.input_size)
