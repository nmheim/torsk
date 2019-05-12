import time
import pathlib
import logging
import numpy as np
from tqdm import tqdm

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

def esn_predict(model, inputs, labels, steps):
    zero_state = np.zeros(model.esn_cell.hidden_size)
    _, states = model.forward(inputs, zero_state, states_only=True)
    pred, _ = model.predict(labels[-1], states[-1], nr_predictions=steps)
    return pred

####################  ESN   ####################################################

params = torsk.Params()
hidden_size = 512
params.input_map_specs = [
    {"type": "random_weights", "size": [hidden_size], "input_scale": 1.}
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
params.train_method = "pinv_svd"
params.tikhonov_beta = 2.0
params.debug = False
params.imed_loss = False

output_dir = pathlib.Path(f"esn_output_{hidden_size}")
model = ESN(params)
model_path = output_dir / "model.pkl"

mackey_train, mackey_eval, mackey_test = mackey_train_eval_test()

dataset = NumpyImageDataset(mackey_train[:, None, None], params)

if not model_path.exists():
    dataset = NumpyImageDataset(mackey_train[:, None, None], params)
    t1 = time.time()
    torsk.train_esn(model, dataset, outdir=output_dir)
    t2 = time.time()
    print(f"ESN Training Time: {t2-t1} s")
else:
    model = torsk.load_model(output_dir)

params.train_length = 100
params.transient_length = 100
params.pred_length = 300
dataset = NumpyImageDataset(mackey_test[:, None, None], params)
inputs, _, _ = dataset[0]

esn_error_path = output_dir / "esn_error.npy"
if not esn_error_path.exists():
    esn_error = []
    print("Generating ESN predictions")
    for inputs, labels, pred_labels in tqdm(dataset):
        zero_state = np.zeros(model.esn_cell.hidden_size)
        _, states = model.forward(inputs, zero_state, states_only=True)
        pred, _ = model.predict(
            labels[-1], states[-1],
            nr_predictions=params.pred_length)
        err = np.abs(pred - pred_labels)
        esn_error.append(err.squeeze())
    
    esn_error = np.mean(esn_error, axis=0)
    np.save(esn_error_path, esn_error)
else:
    esn_error = np.load(esn_error_path)
    

####################  LSTM  ####################################################

import torch
from train import get_data_loaders

hp = torsk.Params()
hp.dtype = "float32"
hp.train_length = 100
hp.pred_length = 300
hp.batch_size = 32
hp.hidden_size = 512
output_dir = f"lstm_output_h{hp.hidden_size}"
 
lstm_model = LSTM(1, hp.hidden_size)
lstm_model.load_state_dict(torch.load(f"{output_dir}/lstm_model_7.pth"))

_, _, loader = get_data_loaders(hp)
inputs, labels, pred_labels = next(iter(loader))

print("Generating LSTM predictions")
lstm_pred = lstm_model.predict(inputs, steps=hp.pred_length)
lstm_pred = lstm_pred.detach().squeeze().numpy()
pred_labels = pred_labels.detach().squeeze().numpy()
lstm_error = np.abs(lstm_pred - pred_labels).mean(axis=0)

esn_inputs = inputs[0].numpy().astype(np.float64)
esn_labels = labels[0].numpy().astype(np.float64)
esn_pred = esn_predict(model, esn_inputs, esn_labels, steps=300).squeeze()

####################  PLOT  ####################################################

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fig, ax = plt.subplots(2,1)
ax[0].plot(pred_labels[0], label="Truth")
ax[0].plot(lstm_pred[0], label="LSTM")
ax[0].plot(esn_pred, label="ESN")
ax[1].plot(lstm_error, color="C1")
ax[1].plot(esn_error, color="C2")
ax[0].legend()
plt.show()
