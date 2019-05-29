import pathlib
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc

import torsk
from torsk.scripts.pred_perf import sort_filenames
from torsk.imed import imed_metric
from torsk.models.torch_lstm import LSTM
from train_esn_lstm_1dmackey import esn_params, lstm_params
from train_esn_lstm_1dmackey import get_data_loaders, mackey_train_eval_test


def generate_cycle_pred(online_esn_dir):
    cycle_preds = []
    for cpredfile in online_esn_dir.glob("cycle_pred_data_idx*.npy"):
        cpred = np.load(cpredfile)
        cycle_preds.append(cpred[:,0,0])
    return np.array(cycle_preds)


def read_online_esn(outdir):
    pred_data_nc_files = list(outdir.glob("pred_data_idx*.nc"))
    esn_error, labels = [], []
    for pred_data_nc in pred_data_nc_files:
        with nc.Dataset(pred_data_nc, 'r') as src:
            pred = src["outputs"][:,0,0]
            lbls = src["labels"][:,0,0]
            labels.append(lbls)
            err = np.abs(pred - lbls)
            esn_error.append(err)
    esn_error = np.array(esn_error).mean(axis=0)
    return  esn_error, np.array(labels)

def generate_offline_esn_pred(inputs_batch, labels_batch, pred_labels_batch, outdir, hidden_size=512):
    model = torsk.load_model(outdir)
    params = esn_params(hidden_size)
    params.train_length = 200
    esn_error = []
    print("Generating ESN predictions")
    for inputs, labels, pred_labels in zip(inputs_batch, labels_batch, pred_labels_batch):
        
        zero_state = np.zeros(model.esn_cell.hidden_size)
        _, states = model.forward(inputs, zero_state, states_only=True)
        pred, _ = model.predict(
            labels[-1], states[-1],
            nr_predictions=params.pred_length)
        err = np.abs(pred - pred_labels)
        esn_error.append(err.squeeze())
    
    esn_error = np.mean(esn_error, axis=0)
    np.save(outdir / "esn_error.npy", esn_error)
    np.save(outdir / "esn_pred.npy", pred)
    np.save(outdir / "esn_lbls.npy", pred_labels)

    inputs = inputs_batch[0]
    labels = labels_batch[0]
    pred_labels = pred_labels_batch[0]
    zero_state = np.zeros(model.esn_cell.hidden_size)
    _, states = model.forward(inputs, zero_state, states_only=True)
    pred, _ = model.predict(
        labels[-1], states[-1],
        nr_predictions=params.pred_length)

    return esn_error, pred, pred_labels

def generate_lstm_preds(inputs, labels, pred_labels, modelpath, hidden_size=512):
    hp = lstm_params(hidden_size=hidden_size)
     
    lstm_model = LSTM(1, hp.hidden_size)
    lstm_model.load_state_dict(torch.load(modelpath))

    print("Generating LSTM predictions")
    lstm_pred = lstm_model.predict(inputs, steps=hp.pred_length)
    lstm_pred = lstm_pred.detach().squeeze().numpy()
    pred_labels = pred_labels.detach().squeeze().numpy()
    lstm_error = np.abs(lstm_pred - pred_labels).mean(axis=0)
    return lstm_error, lstm_pred, pred_labels
    

if __name__ == "__main__":

    output_figure_path = "/home/niklas/erda_save/perf.pdf"
    outdir = pathlib.Path("/mnt/data/torsk_experiments/mackey_1d_esn_lstm")
    offline_esn_dir = outdir / "offline_esn"
    online_esn_dir = outdir / "online_esn"
    lstm_dir = outdir / "lstm"
    hidden_size = 1000

    hp = lstm_params(hidden_size)
    _, _, loader = get_data_loaders(hp)
    inputs, labels, pred_labels = next(iter(loader))

    online_esn_error, online_esn_labels = read_online_esn(online_esn_dir)

    cycle_preds = generate_cycle_pred(online_esn_dir)
    cycle_error = np.abs(online_esn_labels - cycle_preds).mean(axis=0)

    offline_esn_error, esn_pred, _ = generate_offline_esn_pred(
        inputs.numpy().astype(np.float64),
        labels.numpy().astype(np.float64),
        pred_labels.numpy().astype(np.float64),
        offline_esn_dir, hidden_size=hidden_size)

    lstm_error, lstm_pred, pred_labels = generate_lstm_preds(
        inputs, labels, pred_labels, lstm_dir / "lstm_model_20.pth", hidden_size=hidden_size)

    trivial_error = np.abs(pred_labels - pred_labels[:,0][:,None])

    sns.set_style("whitegrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(2,1,figsize=(6, 3.8))
    ax[0].plot(pred_labels[0], label="Truth", color="black")
    ax[0].plot(lstm_pred[0], "--", label="LSTM", color="C3")
    ax[0].plot(esn_pred, ".-", label="ESN (trained once)", color="C0")
    
    step = 5
    t = np.arange(0, hp.pred_length, step)
    ax[1].plot(t, trivial_error[:,::step].mean(axis=0), ":", label="Trivial", color="C2")
    ax[1].plot(t, lstm_error[::step], "--", color="C3", label="LSTM")
    ax[1].plot(t, cycle_error[::step], "-.", label="Cycle-based", color="C1")
    ax[1].plot(t, offline_esn_error[::step], ".-", label="ESN (trained once)", color="C0")
    ax[1].plot(t, online_esn_error[::step], label="ESN (trained online)", color="C0")
    ax[1].set_yscale("log")
    
    ax[0].annotate('A', xy=(0.05, 0.8), xycoords='axes fraction',
        bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})
    ax[1].annotate('B', xy=(0.05, 0.8), xycoords='axes fraction',
        bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})
    
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="lower right")
    fig.savefig(output_figure_path)
