import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import NetcdfDataset, SeqDataLoader
from torsk.visualize import animate_double_imshow


#train_length = 800
#pred_length = 300
#transient_length = 100

params = torsk.Params("params.json")
print(params)

print("Loading + resampling of kuro window ...")
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = NetcdfDataset(ncpath, params.train_length, params.pred_length,
    xslice=slice(90, 190), yslice=slice(90, 190), size=[30, 30])

loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))
#inputs, labels, pred_labels = next(iter(loader))

print("Building model ...")
model = ESN(params)

print("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(model, loader, params)

#print("Training model ...")
#state = torch.zeros(1, params.hidden_size)
#_, states = model(inputs, state)
#model.train(states[transient_length:, 0], labels[transient_length:, 0])
#
#print("Creating prediction ...")
#inputs, labels = next(iter(loader))
#state = torch.zeros(1, params.hidden_size)
#
#warmup_length = 500
#outputs, _ = model(inputs[:warmup_length], state, nr_predictions=labels.shape[0]-warmup_length)

weight = model.esn_cell.res_weight._values().numpy()

hist, bins = np.histogram(weight)
plt.plot(bins[1:], hist)
plt.show()

labels = pred_labels.numpy().reshape([-1, 30, 30])
outputs = outputs.numpy().reshape([-1, 30, 30])
anim = animate_double_imshow(labels, outputs)
plt.show()

