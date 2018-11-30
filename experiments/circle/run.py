import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import CircleDataset, SeqDataLoader
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

Nx, Ny = 80, 80

params = torsk.Params("params.json")
params.input_size  = Nx*Ny
params.output_size = Nx*Ny
logger.info(params)

logger.info("Creating circle dataset")

x = np.sin(np.arange(0, 200*np.pi, 0.1))
y = np.cos(np.arange(0, 200*np.pi, 0.1))
center = np.array([y, x]).T
sigma = 0.4


dataset = CircleDataset(
    params.train_length, params.pred_length,
    center=center, sigma=sigma, size=[Ny, Nx])

loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels, _ = torsk.train_predict_esn(model, loader, params)

weight = model.esn_cell.res_weight._values().numpy()

hist, bins = np.histogram(weight, bins=100)
plt.plot(bins[1:], hist)
plt.show()

labels = pred_labels.numpy().reshape([-1, Nx, Ny])
outputs = outputs.numpy().reshape([-1, Nx, Ny])
anim = animate_double_imshow(labels, outputs)
plt.show()

