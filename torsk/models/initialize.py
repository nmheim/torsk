import numpy as np
from scipy import sparse
from scipy.stats import uniform


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
        eig = np.linalg.eigvals(res)
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
