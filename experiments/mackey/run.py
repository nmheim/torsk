import numpy as np
import matplotlib.pyplot as plt
import torch

from torsk.models import ESN
from torsk.utils import Params
from torsk.data import MackeyDataset, SeqDataLoader


params = Params("params.json")

train_length = params.train_length
pred_length = params.pred_length
transient_length = params.transient_length
beta = params.tikhonov_beta
method = params.train_method
print(params)

# input/label setup
dataset = MackeyDataset(train_length + pred_length, simulation_steps=3000)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

model = ESN(params)

for i in range(10):

    inputs, labels = next(loader)

    train_inputs, train_labels = inputs[:train_length], labels[:train_length]
    test_inputs = inputs[train_length - transient_length:train_length]
    test_labels = labels[train_length - transient_length:]

    # create states and train
    state = torch.zeros(1, params.hidden_size)

    _, states = model(train_inputs, state)
    
    model.train(
        inputs=train_inputs[transient_length:, 0],
        states=states[transient_length:, 0],
        labels=train_labels[transient_length:, 0],
        method=method,
        beta=beta)
    
    # predict
    state = torch.zeros(1, params.hidden_size)
    outputs, _ = model(test_inputs, state, nr_predictions=pred_length)

    err = (test_labels[transient_length:] - outputs[transient_length:])**2
    mse = torch.mean(err)
    print(f"RMSE: {mse.item()}")
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(test_labels.numpy()[transient_length:, 0, 0])
    ax[0].plot(outputs.numpy()[transient_length:, 0, 0])
    ax[0].set_ylim(-0.1, 1.1)
    hist, bins = np.histogram(model.out.weight.numpy(), bins=100)
    ax[1].plot(bins[:-1], hist)
    plt.show()
