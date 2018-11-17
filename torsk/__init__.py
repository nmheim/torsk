import torch
from torsk.models import esn


def mse(predictions, labels):
    err = (predictions - labels)**2
    return torch.mean(err).item()


def train_esn(model, inputs, labels, params):
    tlen = params.transient_length

    zero_state = torch.zeros(1, params.hidden_size)
    _, states = model(inputs, zero_state)

    model.train(
        inputs=inputs[tlen:, 0],
        states=states[tlen:, 0],
        labels=labels[tlen:, 0],
        method=params.train_method,
        beta=params.tikhonov_beta)
    return model, states
