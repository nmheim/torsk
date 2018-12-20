import torch
from torsk.models.numpy_optimize import _pseudo_inverse_svd


def _extended_states(inputs, states):
    ones = torch.ones([inputs.size(0), 1])
    return torch.cat([ones, inputs, states], dim=1).t()


def pseudo_inverse(inputs, states, labels):
    wout = _pseudo_inverse_svd(inputs.numpy(), states.numpy(), labels.numpy())
    return torch.tensor(wout, dtype=inputs.dtype)


def tikhonov(inputs, states, labels, beta):
    X = _extended_states(inputs, states)

    Id = torch.eye(X.size(0))
    A = torch.mm(X, X.t()) + beta * Id
    B = torch.mm(X, labels)

    # Solve linear system instead of calculating inverse
    wout, _ = torch.gesv(B, A)
    return wout.t()
