import numpy as np
import matplotlib.pyplot as plt
import torch
from torsk.models import ESN
from torsk.utils import Params


def make_sine(periods=30, N=20):
    dx = 2*np.pi/(N+1);
    x = np.linspace(0, 2*np.pi-dx, N)
    y = np.sin(x)
    y = np.tile(y, periods)
    return y


# Parameters and model initialization
params = Params("params.json")
nr_predictions = params.pred_length
tlen          = params.transient_length
T             = params.train_length

model = ESN(params)

# do it with the ESN class and the pseudo inverse
data = make_sine(periods=100, N=20)
data = torch.tensor(data.reshape((len(data), 1, 1)),dtype=torch.float32)
all_inputs,   all_labels   = data[:-1],           data[1:]
train_inputs, train_labels = all_inputs[:T+tlen], all_labels[:T+tlen]

state0 = torch.zeros(1, params.hidden_size)
_, train_states = model(train_inputs,state0)


model.train(train_inputs[tlen:],train_states[tlen:], train_labels[tlen:],
            method=params.train_method, beta=params.tikhonov_beta)


outputs, _ = model(train_inputs[-1].unsqueeze(0), train_states[-1], nr_predictions=nr_predictions)

print(outputs.shape)

plt.plot(all_labels[tlen+T:tlen+T+nr_predictions].numpy()[:,0], '.')
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
