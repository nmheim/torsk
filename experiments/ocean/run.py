import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from torsk.models import ESN
from torsk.utils import Params
from torsk.data import NetcdfDataset, SeqDataLoader
from torsk.visualize import animate_double_imshow


train_length = 800
transient_length = 100

# input/label setup
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean.nc')
dataset = NetcdfDataset(ncpath, train_length,
    xslice=slice(90, 190), yslice=slice(90, 190), size=[30, 30])

#inputs, _ = dataset[0]
#plt.imshow(inputs[0].reshape((30,30)))
#plt.show()

loader = SeqDataLoader(dataset, batch_size=1, shuffle=True)
inputs, labels = next(iter(loader))

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
state = torch.zeros(1, params.hidden_size)

warmup_length = 500
outputs, _ = model(inputs[:warmup_length], state, nr_predictions=labels.shape[0]-warmup_length)

labels = labels.numpy()[warmup_length:].reshape([-1, 30, 30])
outputs = outputs.numpy()[warmup_length:].reshape([-1, 30, 30])
anim = animate_double_imshow(labels, outputs)
plt.show()
