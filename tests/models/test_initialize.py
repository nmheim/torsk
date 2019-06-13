import numpy as np
import torsk.models.initialize as init


def test_reservoir_initialization():
    dim = 10
    density = 1.
    symmetric = True
    spectral_radius = 1.

    # connection mask
    mask = init.connection_mask(dim, density, symmetric)
    assert mask.sum() == dim * dim
    assert mask[0, -1] == mask[-1, -1]

    # reservoir matrix
    res = init.dense_esn_reservoir(dim, spectral_radius, density, symmetric)
    # check if symmteric
    assert np.all(res.T == res)

    # sparse reservoir matrix
    res = init.sparse_esn_reservoir(dim, spectral_radius, density, symmetric)
    dense = res.data
    assert np.all(dense.T == dense)
