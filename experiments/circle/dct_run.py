import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import DCTCircleDataset, SeqDataLoader
from torsk.data.utils import idct2_sequence
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)


params = torsk.Params("params.json")
Nfrq = 10
params.input_size  = Nfrq**2
params.output_size = Nfrq**2 

params.train_length = 1500
params.pred_length = 600

params.train_method = "tikhonov"
params.tikhonov_beta = 5

logger.info(params)


logger.info("Loading + resampling of input data ...")
x = np.sin(np.arange(0, 200*np.pi, 0.1))
y = np.cos(0.25 * np.arange(0, 200*np.pi, 0.1))
center = np.array([y, x]).T
sigma = 0.2

dataset = DCTCircleDataset(
    params.train_length, params.pred_length,
    center=center, sigma=sigma, xsize=[20, 20], ksize=[Nfrq, Nfrq])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))


logger.info("Building model ...")
model = ESN(params)


logger.info("Training + predicting ...")
model, outputs, pred_labels, original = torsk.train_predict_esn(model, loader, params)


logger.info("Visualizing results ...")
original = torch.squeeze(original).numpy()
weight = model.esn_cell.res_weight._values().numpy()

hist, bins = np.histogram(weight, bins=100)
plt.plot(bins[1:], hist)
plt.show()

outputs = outputs.numpy().reshape([-1, Nfrq, Nfrq])
outputs = idct2_sequence(outputs, xsize=original.shape[1:])

y, x = 10, 10
plt.plot(original[:, y, x])
plt.plot(outputs[:, y, x])
plt.show()

anim = animate_double_imshow(original, outputs.real)
plt.show()
