import torch
from torsk.models.numpy_optimize import pseudo_inverse as np_pinv


def _extended_states(inputs, states):
    ones = torch.ones([inputs.size(0), 1])
    return torch.cat([ones, inputs, states], dim=1).t()


def pseudo_inverse(inputs, states, labels, mode="svd"):
    wout = np_pinv(
        inputs.squeeze(dim=1).numpy(),
        states.squeeze(dim=1).numpy(),
        labels.squeeze(dim=1).numpy(),
        mode=mode)
    return torch.tensor(wout, dtype=inputs.dtype)


def tikhonov(inputs, states, labels, beta):
    train_length = inputs.shape[0]
    flat_inputs = inputs.reshape([train_length, -1])
    flat_labels = labels.reshape([train_length, -1])

    X = _extended_states(flat_inputs, states)

    Id = torch.eye(X.size(0))
    A = torch.mm(X, X.t()) + beta * Id
    B = torch.mm(X, flat_labels)

    # Solve linear system instead of calculating inverse
    wout, _ = torch.gesv(B, A)
    return wout.t()
