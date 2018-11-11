import numpy as np
import matplotlib.pyplot as plt
import torch

from torsk.models import ESN
from torsk.utils import Params
from torsk.data import MackeyDataset, SeqDataLoader


train_length = 2200
transient_length = 200

# input/label setup
dataset = MackeyDataset(train_length, simulation_steps=3000)
loader = SeqDataLoader(dataset, batch_size=1, shuffle=False)
inputs, labels = next(iter(loader))
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# build model
params = Params("params.json")
print(params)
model = ESN(params)

# define initial state
state = torch.zeros(1, params.hidden_size)

# create states and train
_, states = model(inputs, state)
model.train(states[transient_length:, 0], labels[transient_length:, 0])


inputs, labels = next(iter(loader))
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
state = torch.zeros(1, params.hidden_size)

outputs, _ = model(inputs[:1000], state, nr_predictions=1000)

plt.plot(labels.numpy()[1000:1500,0])
plt.plot(outputs.numpy()[1000:1500,0])
plt.show()
