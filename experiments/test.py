import numpy as np
import matplotlib.pyplot as plt
import torch
from torsk.esn import ESNCell
from torsk.utils import Params


class Network(torch.nn.Module):

    def __init__(self, params):
        super(Network, self).__init__()

        self.esn = ESNCell(
            input_size=params.input_size,
            hidden_size=params.hidden_size,
            spectral_radius=params.spectral_radius,
            in_weight_init=params.in_weight_init)
        self.out = torch.nn.Linear(params.hidden, params.output_size)

    def forward(self, inputs):
        outputs = []

        for inp in inputs:
            state = model(inp, state)
            output = self.out(state)
            outputs.append(output)
        for ii in range(100):
            inp = output
            state = model(inp, state)
            output = self.out(state)
            outputs.append(output)
        
        outputs = torch.cat(outputs).numpy()



def make_sine(periods=30):
    x = np.linspace(0, 2*np.pi-np.pi/10, 20)
    y = np.sin(x)
    y = np.tile(y, periods)
    return y

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


# pseudo inverse

data = make_sine(periods=100)
data = data.reshape((data.shape[0], 1, 1))
inputs, labels = data[:-1], data[1:]

inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels[10*20:,0,:].T, dtype=torch.float32)

params = Params("params.json")
model = ESNCell(
    input_size=params.input_size,
    hidden_size=params.hidden_size,
    spectral_radius=params.spectral_radius,
    in_weight_init=params.in_weight_init)

state = torch.zeros(1, params.hidden_size)

states = []
for inp in inputs:
    state = model(inp, state)
    states.append(state)
states = torch.cat(states[10*20:]).t()

pinv = torch.pinverse(states)
wout = torch.mm(labels, pinv)

data = make_sine(periods=10)
data = data.reshape((data.shape[0], 1, 1))
inputs, labels = data[:-1], data[1:]
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
state = torch.zeros(1, params.hidden_size)


outputs = []

for inp in inputs:
    state = model(inp, state)
    # TODO: transpose only works for batch_size = 1 !!!
    output = torch.mm(wout, state.t())
    outputs.append(output)
for ii in range(100):
    inp = output
    state = model(inp, state)
    output = torch.mm(wout, state.t())
    outputs.append(output)

outputs = torch.cat(outputs).numpy()

plt.plot(labels.numpy()[:,0])
plt.plot(outputs)
#plt.ylim(-1, 1)
plt.show()
