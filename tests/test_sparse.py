from torsk.numpy_accelerate import bh
from torsk.sparse import SparseMatrix


def test_square_sparse():

    values = bh.arange(1, 11)
    col_idx = bh.arange(10)
    nonzeros_per_row = 1
    dense_shape = (10, 10)

    dense = bh.eye(10) * values

    mat = SparseMatrix(values, col_idx, nonzeros_per_row, dense_shape)
    dmat = SparseMatrix.from_dense(dense, nonzeros_per_row)
    assert bh.all(mat.values == dmat.values)
    assert bh.all(mat.col_idx == dmat.col_idx)
    assert bh.all(mat.dense_shape == dense_shape)
    assert bh.all(dmat.dense_shape == dense_shape)

    x = bh.ones(10)
    y = mat.sparse_dense_mv(x)
    assert bh.all(y == bh.arange(1, 11))

    X = bh.ones([10, 10])
    Y = mat.sparse_dense_mm(X)
    for ii, y in enumerate(Y):
        assert bh.all(ii + 1 == y)


def test_rect_sparse():

    array = bh.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]])

    mat = SparseMatrix.from_dense(array, nonzeros_per_row=1)

    vec = bh.array([1, 2, 3, 4])

    assert bh.all(mat.sparse_dense_mv(vec) == bh.array([3, 1, 2]))
