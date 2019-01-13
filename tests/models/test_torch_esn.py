import numpy as np
import torch
import torsk
from torsk.models.torch_esn import TorchESN, TorchMapESNCell, TorchMapSparseESNCell


def model_forward(dtype_str, reservoir):
    params = torsk.default_params()
    params.dtype = dtype_str
    params.backend = "torch"
    params.reservoir_representation = reservoir
    model = TorchESN(params)

    torch.set_default_dtype(model.esn_cell.dtype)

    inputs = torch.ones([3] + params.input_shape)
    state = torch.zeros([1, model.esn_cell.hidden_size])

    outputs, states = model(inputs, state, states_only=False)
    return outputs, states


def test_dtypes():
    for dtype_str in ["float32", "float64"]:
        for reservoir in ["sparse", "dense"]:
            outputs, states = model_forward(dtype_str, reservoir)
            dtype = getattr(torch, dtype_str)
            assert outputs.dtype == dtype
            assert states.dtype == dtype


def test_esn_cell():
    input_shape = [10, 12]
    specs = [
        {"type": "pixels", "size": [5, 6], "input_scale": 2},
        # {"type": "dct", "size": [5, 5], "input_scale": 2},  TODO: implement!
        {"type": "conv", "size": [5, 5], "kernel_type": "gauss", "input_scale": 2},
        {"type": "random_weights", "size": [100], "weight_scale": 2}]

    hidden_size = 5 * 6
    # hidden_size += 5 * 5  # dct size
    hidden_size += 6 * 8
    hidden_size += 100
    spectral_radius = 0.5
    density = 1.0
    dtype = "float64"

    dense_cell = TorchMapESNCell(
        input_shape=input_shape,
        input_map_specs=specs,
        spectral_radius=spectral_radius,
        density=density,
        dtype=dtype)

    torch.set_default_dtype(dense_cell.dtype)

    assert dense_cell.weight_hh.shape == (hidden_size, hidden_size)

    inputs = torch.rand(*input_shape)
    state = torch.rand(1, hidden_size)

    new_state = dense_cell.forward(inputs, state)
    assert state.shape == new_state.shape
    assert np.any(state.numpy() != new_state.numpy())
    assert new_state.dtype == dense_cell.dtype

    sparse_cell = TorchMapSparseESNCell(
        input_shape=input_shape,
        input_map_specs=specs,
        spectral_radius=spectral_radius,
        density=density,
        dtype=dtype)

    new_state = sparse_cell.forward(inputs, state)
    assert state.shape == new_state.shape
    assert np.any(state.numpy() != new_state.numpy())


def test_esn():

    # check default parameters
    params = torsk.default_params()
    params.train_method = "tikhonov"
    params.tikhonov_beta = 10

    lag_len = 3
    model = TorchESN(params)
    torch.set_default_dtype(model.esn_cell.dtype)
    inputs = torch.rand([lag_len] + params.input_shape)
    labels = torch.rand([lag_len] + params.input_shape)
    state = torch.rand([1, model.esn_cell.hidden_size])

    # test forward
    outputs, states = model(inputs, state, states_only=False)
    assert outputs.size() == (lag_len,) + tuple(params.input_shape)
    assert states.size() == (lag_len, model.esn_cell.hidden_size)

    # test train
    wout = model.out.weight.detach().numpy()
    model.optimize(inputs=inputs, states=states, labels=labels)
    assert not np.any(wout == model.out.weight.detach().numpy())

    # test predict
    outputs, states = model.predict(inputs[-1], state, nr_predictions=2)
    assert states.size() == (2, model.esn_cell.hidden_size)
    assert outputs.size() == (2,) + tuple(params.input_shape)
