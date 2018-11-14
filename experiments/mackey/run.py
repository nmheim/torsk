import numpy as np
import matplotlib.pyplot as plt
import torch

from torsk.models import ESN
from torsk.utils import Params
from torsk.data import MackeyDataset, SeqDataLoader


train_length = 2200
pred_length = 500
transient_length = 200

# input/label setup
dataset = MackeyDataset(train_length + pred_length, simulation_steps=3000)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

# build model
params = Params("params.json")
print(params)
model = ESN(params)


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
    labels=train_labels[transient_length:, 0])

# predict
state = torch.zeros(1, params.hidden_size)
outputs, _ = model(test_inputs, state, nr_predictions=pred_length)

plt.plot(test_labels.numpy()[transient_length:, 0, 0])
plt.plot(outputs.numpy()[transient_length:, 0, 0])
plt.show()
