import torch
from torsk.models import esn


def mse(predictions, labels):
    err = (predictions - labels)**2
    return torch.mean(err).item()


def train_predict_esn(model, loader, params):
    tlen = params.transient_length

    inputs, labels, pred_labels = next(loader)

    zero_state = torch.zeros(1, params.hidden_size)
    _, states = model(inputs, zero_state)

    model.train(
        inputs=inputs[tlen:],
        states=states[tlen:],
        labels=labels[tlen:],
        method=params.train_method,
        beta=params.tikhonov_beta)

    # predict
    init_inputs = inputs[-1].unsqueeze(0)
    outputs, _ = model(
        init_inputs, states[-1], nr_predictions=params.pred_length - 1)

    return model, outputs, pred_labels
