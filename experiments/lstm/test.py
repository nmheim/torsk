import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt

import torsk
from torsk.data.utils import mackey_sequence
from torsk.data.numpy_dataset import NumpyImageDataset
from torsk.models.numpy_esn import NumpyESN as ESN
from torsk.models.torch_lstm import LSTM
logging.basicConfig(level="INFO")

def mackey_train_eval_test():
    mackey = mackey_sequence(N=4000)
    mackey_train, mackey_eval = mackey[:2500], mackey[2500:3000]
    mackey_test = mackey[3000:]
    return mackey_train, mackey_eval, mackey_test

####################  ESN   ####################################################

params = torsk.Params()
params.input_map_specs = [
    {"type": "random_weights", "size": [1000], "input_scale": 1.}
]
params.spectral_radius = 1.5
params.density = 0.1
params.input_shape = [1, 1]
params.train_length = 2200
params.pred_length = 100
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "dense"
params.backend = "numpy"
params.train_method = "pinv_lstsq"
params.tikhonov_beta = 2.0
params.debug = False
params.imed_loss = False

model = ESN(params)
model_path = pathlib.Path("esn_output/model.pkl")

mackey_train, mackey_eval, mackey_test = mackey_train_eval_test()

if not model_path.exists():
    dataset = NumpyImageDataset(mackey_train[:, None, None], params)
    torsk.train_esn(model, dataset, outdir="esn_output")
else:
    model = torsk.load_model("esn_output")

params.train_length = 100
params.transient_length = 100
params.pred_length = 300
dataset = NumpyImageDataset(mackey_test[:, None, None], params)

esn_error = []
for inputs, labels, pred_labels in dataset:
    zero_state = np.zeros(model.esn_cell.hidden_size)
    _, states = model.forward(inputs, zero_state, states_only=True)
    pred, _ = model.predict(
        labels[-1], states[-1],
        nr_predictions=params.pred_length)
    err = np.abs(pred - pred_labels)
    esn_error.append(err.squeeze())

esn_error = np.mean(esn_error, axis=0)

####################  LSTM  ####################################################

import torch
from train import get_data_loaders

hp = torsk.Params()
hp.dtype = "float32"
hp.train_length = 100
hp.pred_length = 300
hp.batch_size = 32
output_dir = "lstm_output"
 
model = LSTM(1, 128)
model.load_state_dict(torch.load(f"{output_dir}/lstm_model_1.pth"))

_, _, loader = get_data_loaders(hp)
inputs, labels, pred_labels = next(iter(loader))

pred = model.predict(inputs, steps=hp.pred_length)
pred = pred.detach().squeeze().numpy()
pred_labels = pred_labels.detach().squeeze().numpy()
lstm_error = np.abs(pred - pred_labels).mean(axis=0)

####################  PLOT  ####################################################

fig, ax = plt.subplots(2,1)
ax[0].plot(pred)
ax[0].plot(pred_labels)
ax[1].plot(lstm_error)
ax[1].plot(esn_error)
plt.show()


