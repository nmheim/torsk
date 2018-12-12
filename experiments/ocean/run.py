import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import NetcdfDataset, SeqDataLoader
from torsk.visualize import animate_double_imshow, write_video


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

Nx, Ny = 30, 30

params = torsk.Params("params.json")
params.input_size  = Nx*Ny
params.output_size = Nx*Ny
params.train_length = 1500
params.train_method = "tikhonov"
params.tikhonov_beta = 3e1
logger.info(params)

logger.info("Loading + resampling of kuro window ...")
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = NetcdfDataset(ncpath, params.train_length, params.pred_length,
    xslice=slice(90, 190), yslice=slice(90, 190), size=[Nx, Ny])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels, _ = torsk.train_predict_esn(model, loader, params)

# weight = model.esn_cell.res_weight._values().numpy()
# 
# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

labels = pred_labels.numpy().reshape([-1, Nx, Ny])
outputs = outputs.numpy().reshape([-1, Nx, Ny])

y, x = 10, 10
plt.plot(labels[:, y, x])
plt.plot(outputs[:, y, x])
plt.show()

anim = animate_double_imshow(
    labels,# labels[params.train_length - 50: params.train_length + 50],
    outputs)#outputs[params.train_length - 50: params.train_length + 50])
plt.show()

