import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import DCTCircleDataset, SeqDataLoader
from torsk.data.utils import idct2
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

Nfrq = 20
params = torsk.Params("dct_params.json")
params.input_size  = Nfrq**2
params.output_size = Nfrq**2 

logger.info(params)

logger.info("Loading + resampling of kuro window ...")
x = np.sin(np.arange(0, 200*np.pi, 0.1))
y = np.cos(0.5 * np.arange(0, 200*np.pi, 0.1))
center = np.array([y, x]).T
sigma = 0.2

dataset = DCTCircleDataset(
    params.train_length, params.pred_length,
    center=center, sigma=sigma, size=[100, 100], resize=[Nfrq, Nfrq])

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

