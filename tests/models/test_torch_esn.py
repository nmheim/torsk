import numpy as np
import torch
import torsk
from torsk.models.torch_esn import TorchESN, TorchESNCell, TorchSparseESNCell


def model_forward(dtype_str, reservoir):
    params = torsk.default_params()
    params.dtype = dtype_str
    params.backend = "torch"
    params.reservoir_representation = reservoir
    model = TorchESN(params)

    torch.set_default_dtype(model.esn_cell.dtype)

    inputs = torch.ones([3, 1, params.input_size])
    state = torch.zeros([1, params.hidden_size])

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
    input_size = 1
    hidden_size = 10
    spectral_radius = 0.5
    weight_init = 0.1
    bias_init = 0.1
    batch_size = 3
    density = 1.0

    dense_cell = TorchESNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        spectral_radius=spectral_radius,
        in_weight_init=weight_init,
        in_bias_init=bias_init,
        density=density,
        dtype="float64")

    assert dense_cell.weight_hh.size() == (hidden_size, hidden_size)
    assert not dense_cell.weight_hh.requires_grad
    assert dense_cell.weight_ih.size() == (hidden_size, input_size)
    assert not dense_cell.weight_ih.requires_grad

    torch.set_default_dtype(dense_cell.dtype)
    inputs = torch.rand([batch_size, input_size])
    state = torch.rand(batch_size, hidden_size)

    new_state = dense_cell(inputs, state)
    assert state.size() == new_state.size()
    assert np.any(state.numpy() != new_state.numpy())

    sparse_cell = TorchSparseESNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        spectral_radius=spectral_radius,
        in_weight_init=weight_init,
        in_bias_init=bias_init,
        density=density,
        dtype="float64")

    inputs = torch.rand([1, input_size])
    state = torch.rand(1, hidden_size)

    new_state = sparse_cell(inputs, state)
    assert state.size() == new_state.size()
    assert np.any(state.numpy() != new_state.numpy())


def test_esn():

    # check default parameters
    params = torsk.default_params()
    params.input_size = 1
    params.hidden_size = 100
    params.output_size = 1

    # check model
    lag_len = 3
    batch_size = 1
    model = TorchESN(params)
    torch.set_default_dtype(model.esn_cell.dtype)
    inputs = torch.rand([lag_len, batch_size, params.input_size])
    labels = torch.rand([lag_len, params.output_size, params.output_size])
    state = torch.rand([batch_size, params.hidden_size])

    # test _forward_states_only
    outputs, states = model(inputs, state, states_only=False)
    assert outputs.size() == (lag_len, batch_size, params.output_size)
    assert states.size() == (lag_len, batch_size, params.hidden_size)

    # test train
    wout = model.out.weight.detach().numpy()
    model.optimize(inputs=inputs, states=states, labels=labels)
    assert not np.any(wout == model.out.weight.detach().numpy())

    # test _forward
    outputs, states = model.predict(inputs[-1], state, nr_predictions=2)
    assert states.size() == (2, batch_size, params.hidden_size)
    assert outputs.size() == (2, batch_size, params.output_size)
