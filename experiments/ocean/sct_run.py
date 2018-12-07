import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import SCTNetcdfDataset, SeqDataLoader
from torsk.data.utils import dct2, idct2, isct2, sct2, dct2_sequence, idct2_sequence
from torsk.visualize import animate_double_imshow


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

#region = [[90,190],[90,190]];
#region = [[200,300],[100,200]];
#nk1,nk2=48,73;
nk1,nk2=48,73;
CTmethod="SCT";

params = torsk.Params("fft_params.json")
params.input_size  = nk1*nk2
params.output_size = nk1*nk2

logger.info(params)

logger.info("Loading + resampling of kuro window ...")
ncpath = pathlib.Path(f'../../data/ocean/kuro_SSH_3daymean_{CTmethod}_{nk1}_{nk2}.nc')
dataset = SCTNetcdfDataset(
    ncpath,
    params.train_length,
    params.pred_length)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))
#writer = dataset.open_output(f"output/kuro_SSH_3daymean_{CTmethod}_{nk1}x{nk2}.nc");

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels, ssh = torsk.train_predict_esn(model, loader, params)

ssh = torch.squeeze(ssh)
weight = model.esn_cell.res_weight._values().numpy()

# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

#        o_full_mask  = nout.createVariable("full_mask",'u1',("nlat","nlon"));
outputs = dataset.to_image(outputs.numpy())
anim = animate_double_imshow(ssh, outputs.real)
plt.show()

