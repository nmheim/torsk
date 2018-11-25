import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torsk
from torsk.models import ESN
from torsk.data import FFTNetcdfDataset, SeqDataLoader
from torsk.data.ocean import ifft
from torsk.visualize import animate_double_imshow

params = torsk.Params("fft_params.json")
print(params)

print("Loading + resampling of kuro window ...")
ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = FFTNetcdfDataset(ncpath, params.train_length, params.pred_length,
    xslice=slice(90, 190), yslice=slice(90, 190), size=[15, 15])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

print("Building model ...")
model = ESN(params)

print("Training + predicting ...")
def train_predict_esn(model, loader, params):
    tlen = params.transient_length

    inputs, labels, pred_labels, ssh = next(loader)

    zero_state = torch.zeros(1, params.hidden_size)
    _, states = model(inputs, zero_state)

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

outputs = outputs.numpy()
outputs_real, outputs_imag = outputs[:, 0, :225], outputs[:, 0, 225:]
outputs_real = outputs_real.reshape([-1, 15, 15])
outputs_imag = outputs_imag.reshape([-1, 15, 15])
outputs = outputs_real + outputs_imag * 1j

outputs = ifft(outputs, size=[100, 100])
anim = animate_double_imshow(ssh, outputs.real)
plt.show()

