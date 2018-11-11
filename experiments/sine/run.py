import numpy as np
import matplotlib.pyplot as plt
import torch
from torsk.models import ESN
from torsk.utils import Params


def make_sine(periods=30):
    x = np.linspace(0, 2*np.pi-np.pi/10, 20)
    y = np.sin(x)
    y = np.tile(y, periods)
    return y


# do it with the ESN class and the pseudo inverse
data = make_sine(periods=100)
data = data.reshape((data.shape[0], 1, 1))
inputs, labels = data[:-1], data[1:]

inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels[:,0,:], dtype=torch.float32)

params = Params("params.json")
model = ESN(params)
state = torch.zeros(1, params.hidden_size)

_, states = model(inputs, state)
model.train(states[10*20:, 0], labels[10*20:])

data = make_sine(periods=10)
data = data.reshape((data.shape[0], 1, 1))
inputs, labels = data[:-1], data[1:]
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
state = torch.zeros(1, params.hidden_size)
outputs, _ = model(inputs, state, nr_predictions=100)

plt.plot(labels.numpy()[:,0], '.')
plt.plot(outputs.numpy()[:,0])
#plt.ylim(-1, 1)
plt.show()

# do it with GD

#data = make_sine(periods=11)
#data = data.reshape(11, 20).T
#data = data.reshape(20, 11, 1)
#inputs, labels = data[:-1], data[1:]
#
#inputs = torch.tensor(inputs, dtype=torch.float32)
#labels = torch.tensor(labels, dtype=torch.float32)
#state = torch.zeros(1, params.hidden_size)
#
#criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(lr=1e-3)
#params = Params("params.json")
#model = ESNCell(
#    input_size=params.input_size,
#    hidden_size=params.hidden_size,
#    spectral_radius=params.spectral_radius,
#    in_weight_init=params.in_weight_init)
#
#for ii in range(100):
#
#    states = []
#    for inp in inputs:
#        state = model(inp, state)
#        states.append(state)
