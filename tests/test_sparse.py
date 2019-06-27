import numpy as np
from torsk.sparse import SparseMatrix


def test_square_sparse():

    values = np.arange(1, 11)
    row_idx = np.arange(10)
    col_idx = np.arange(10)
    nonzeros_per_row = 1
    dense_shape = (10, 10)

    dense = np.eye(10) * values

    mat = SparseMatrix(values, col_idx, nonzeros_per_row, dense_shape)
    dmat = SparseMatrix.from_dense(dense)
    assert np.all(mat.values == dmat.values)
    assert np.all(mat.row_idx == dmat.row_idx)
    assert np.all(mat.col_idx == dmat.col_idx)
    assert np.all(mat.dense_shape == dense_shape)
    assert np.all(dmat.dense_shape == dense_shape)

    X = np.ones([10, 10])
    Y = mat.sparse_dense_mm(X)
    for ii, y in enumerate(Y):
        assert np.all(ii + 1 == y)

    x = np.ones(10)
    y = mat.sparse_dense_mv(x)
    assert np.all(y == np.arange(1, 11))


def test_rect_sparse():

    array = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]])

    mat = SparseMatrix.from_dense(array, nonzeros_per_row=1)

    vec = np.array([1, 2, 3, 4])

    assert np.all(mat.sparse_dense_mv(vec) == np.array([3, 1, 2]))
