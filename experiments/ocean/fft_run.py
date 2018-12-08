import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import DCTNetcdfDataset, SeqDataLoader
from torsk.data.utils import idct2_sequence
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

region = [[90,190],[90,190]]  # kuroshio
region_size = [region[0][1] - region[0][0], region[1][1] - region[1][0]]
# region = [[200,300],[100,200]]  # no land
# region = [[0,300],[0,200]]
Nfrq = 40
params = torsk.Params("fft_params.json")
params.input_size  = Nfrq**2
params.output_size = Nfrq**2 
params.hidden_size = 10000
params.pred_length = 300
params.train_length = 1500
params.train_method = "tikhonov"
params.tikhonov_beta = 1e2

logger.info(params)

logger.info("Loading + resampling of kuro window ...")
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = DCTNetcdfDataset(
    ncpath,
    params.train_length,
    params.pred_length,
    xslice=slice(region[0][0], region[0][1]),
    yslice=slice(region[1][0], region[1][1]),
    size=[Nfrq, Nfrq])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, labels, ssh = torsk.train_predict_esn(model, loader, params)

ssh = torch.squeeze(ssh).numpy()
weight = model.esn_cell.res_weight._values().numpy()

# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

outputs = outputs.numpy().reshape([-1, Nfrq, Nfrq])
outputs = idct2_sequence(outputs, region_size)

y, x = 30, 30
plt.plot(ssh[:, y, x])
plt.plot(outputs[:, y, x])
plt.show()

anim = animate_double_imshow(ssh, outputs.real)
plt.show()

