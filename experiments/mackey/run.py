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
loader = SeqDataLoader(dataset, batch_size=1, shuffle=True)
inputs, labels = next(iter(loader))
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

train_inputs, train_labels = inputs[:train_length], labels[:train_length]

# build model
params = Params("params.json")
print(params)
model = ESN(params)

# define initial state
state = torch.zeros(1, params.hidden_size)

# create states and train
_, states = model(train_inputs, state)
model.train(states[transient_length:, 0], train_labels[transient_length:, 0])


#inputs, labels = next(iter(loader))
test_inputs = inputs[train_length - transient_length:train_length]
test_labels = labels[train_length - transient_length:]
state = torch.zeros(1, params.hidden_size)

outputs, _ = model(test_inputs, state, nr_predictions=pred_length)
print(test_inputs.shape)
print(test_labels.shape)
print(outputs.shape)

plt.plot(test_labels.numpy()[transient_length:, 0, 0])
plt.plot(outputs.numpy()[transient_length:, 0, 0])
plt.show()
