import numpy as np
import scipy as sp


def pseudo_inverse_svd(inputs, states, labels):
    X = _extended_states(inputs, states)

    U, s, Vh = sp.linalg.svd(X)
    L = labels.T
    condition = s[0] / s[-1]

    scale = s[0]
    n = len(s[np.abs(s / scale) > 1e-4])  # Ensure condition number less than 10.000
    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = np.dot(np.dot(L, v) * (1 / s[:n]), uh)
    return wout
    

def pseudo_inverse(inputs, states, labels):
    return pseudo_inverse_svd(inputs, states, labels)


def _extended_states(inputs, states):
    ones = np.ones([inputs.shape[0], 1])
    return np.concatenate([ones, inputs, states], axis=1).T
