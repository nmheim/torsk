import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import FFTNetcdfDataset, SeqDataLoader
from torsk.data.ocean import iDCT
from torsk.visualize import animate_double_imshow

#region = [[90,190],[90,190]];
#region = [[200,300],[100,200]];
region = [[0,300],[0,200]]
Nfrq = 110
params = torsk.Params("fft_params.json")
params.input_size  = Nfrq**2
params.output_size = Nfrq**2 
print(params)

print("Loading + resampling of kuro window ...")
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = FFTNetcdfDataset(ncpath, params.train_length, params.pred_length,
    xslice=slice(region[0][0],region[0][1]), yslice=slice(region[1][0], region[1][1]), size=[Nfrq, Nfrq])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

print("Building model ...")
model = ESN(params)

print("Training + predicting ...")
def train_predict_esn(model, loader, params):
    tlen = params.transient_length

    print("Loading data")
    inputs, labels, pred_labels, ssh = next(loader)

    zero_state = torch.zeros(1, params.hidden_size)
    print("Initializing model")
    _, states = model(inputs, zero_state)

    print("Training the model")
    model.train(
        inputs=inputs[tlen:],
        states=states[tlen:],
        labels=labels[tlen:],
        method=params.train_method,
        beta=params.tikhonov_beta)

    # predict
    init_inputs = inputs[-1].unsqueeze(0)
    outputs, _ = model(
        init_inputs, states[-1], nr_predictions=params.pred_length - 1)

    return model, outputs, np.squeeze(ssh)

model, outputs, ssh = train_predict_esn(model, loader, params)

weight = model.esn_cell.res_weight._values().numpy()

# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

outputs = outputs.numpy().reshape([-1, Nfrq, Nfrq])

outputs = iDCT(outputs, size=ssh.shape[1:])
anim = animate_double_imshow(ssh, outputs.real)
plt.show()

