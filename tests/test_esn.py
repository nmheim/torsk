import numpy as np
import torch
from torsk import esn


def test_reservoir_initialization():
    dim = 10
    density = 1.
    symmetric = True
    spectral_radius = 1.

    # connection mask
    mask = esn.connection_mask(dim, density, symmetric)
    assert mask.sum() == dim * dim
    assert mask[0, -1] == mask[-1, -1]
    
    # reservoir matrix
    res = esn.dense_esn_reservoir(dim, spectral_radius, density, symmetric)
    # check if symmteric
    assert np.all(res.T == res)


def test_esn_cell():
    input_size = 1
    hidden_size = 10
    spectral_radius = 0.5
    weight_init = 0.1
    bias_init = 0.1
    batch_size = 3
    density = 1.0
        
    cell = esn.ESNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        spectral_radius=spectral_radius,
        in_weight_init=weight_init,
        in_bias_init=bias_init,
        density=density)
    assert cell.res_weight.size() == (hidden_size, hidden_size)
    assert cell.res_weight.requires_grad == False
    assert cell.in_weight.size() == (hidden_size, input_size)
    assert cell.in_weight.requires_grad == False

    inputs = torch.Tensor(np.random.uniform(size=[batch_size, input_size]))
    state = torch.Tensor(np.random.uniform(size=[batch_size, hidden_size]))
    new_state = cell(inputs, state)
    assert state.size() == new_state.size()
    assert np.any(state.numpy() != new_state.numpy())


def test_esn():

    # check default parameters
    params = esn.get_default_params()
    assert params.input_size == 1
    assert params.hidden_size == 50
    assert params.output_size == 1
    assert params.spectral_radius == 1.1
    assert params.in_weight_init == 0.5
    assert params.density == 1.0

    # check model
    lag_len = 3
    batch_size = 1
    model = esn.ESN(params)
    inputs = torch.rand([lag_len, batch_size, params.input_size])
    state = torch.rand([batch_size, params.hidden_size])

    # test _forward_states_only
    outputs, states = model(inputs, state, nr_predictions=0)
    assert outputs is None
    assert states.size() == (lag_len, batch_size, params.hidden_size)

    # test train
    wout = model.out.weight.detach().numpy()
    model.train(
        torch.squeeze(states),
        torch.rand([lag_len, params.output_size]))
    assert not np.any(wout == model.out.weight.detach().numpy())

    # test _forward
    outputs, states = model(inputs, state, nr_predictions=2)
    assert states is None
    assert outputs.size() == (lag_len + 2, batch_size, params.output_size)
