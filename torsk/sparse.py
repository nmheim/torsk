import numpy as np


def sparse_dense_mm(A, X):
    """Matrix-matrix multiplication for fixed nonzero per row sparse matrix"""
    assert A.dense_shape[1] == X.shape[0]
    AXv = A.values[:, None] * X[A.col_idx, :]
    return np.sum(AXv.reshape((-1, X.shape[1], A.nonzeros_per_row)), axis=2)


def sparse_dense_mv(A, x):
    """Matrix-vector multiplication for fixed nonzero per row sparse matrix"""
    assert A.dense_shape[1] == x.shape[0]
    Axv = A.values * x[A.col_idx]
    return np.sum(Axv.reshape(-1, A.nonzeros_per_row), axis=1)


class SparseMatrix:
    """A simple COO sparse matrix implementation that allows only matrices
    with a fixed number of nonzero elements per row.
    """
    def __init__(self, values, row_idx, col_idx, nonzeros_per_row, dense_shape):
        assert values.shape == row_idx.shape == col_idx.shape

        self.values = values
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.nonzeros_per_row = nonzeros_per_row
        self.dense_shape = dense_shape

        self._check_nonzeros()

    def _check_nonzeros(self):
        for ii in range(self.dense_shape[1]):
            idx = self.row_idx == ii
            nzv = np.sum(self.values[idx] > 0)
            assert nzv == self.nonzeros_per_row

    @classmethod
    def from_dense(cls, dense_matrix, nonzeros_per_row):
        row_idx, col_idx = np.nonzero(dense_matrix)
        values = dense_matrix[col_idx, row_idx]
        return cls(values, row_idx, col_idx, nonzeros_per_row, dense_matrix.shape)

    def sparse_dense_mm(self, X):
        return sparse_dense_mm(self, X)

    def sparse_dense_mv(self, x):
        return sparse_dense_mv(self, x)

    def __str__(self):
        nz = self.nonzeros_per_row
        ds = self.dense_shape
        name = self.__class__.__name__
        string = f"<{name} nonzeros_per_row:{nz} values:{self.values} shape:{ds}>"
        return string


def random_sparse_matrix(dense_shape, low=-1, high=1):
    pass
