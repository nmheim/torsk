import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import DCTNetcdfDataset, SeqDataLoader
from torsk.data.ocean import idct2
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

#region = [[90,190],[90,190]];
#region = [[200,300],[100,200]];
region = [[0,300],[0,200]]
Nfrq = 110
params = torsk.Params("fft_params.json")
params.input_size  = Nfrq**2
params.output_size = Nfrq**2 

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
model, outputs, pred_labels, ssh = torsk.train_predict_esn(model, loader, params)

ssh = torch.squeeze(ssh)
weight = model.esn_cell.res_weight._values().numpy()

# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

outputs = outputs.numpy().reshape([-1, Nfrq, Nfrq])

outputs = idct2(outputs, size=ssh.shape[1:])
anim = animate_double_imshow(ssh, outputs.real)
plt.show()

