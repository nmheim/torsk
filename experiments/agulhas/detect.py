import pathlib
from itertools import cycle
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import torsk
from torsk.models.torch_lstm import LSTM
from torsk.scripts.pred_perf import sort_filenames
from torsk.data.utils import mackey_anomaly_sequence, normalize
from torsk.imed import imed_metric
from torsk.anomaly import sliding_score
from torsk import Params
from utils import initialize, finalize

args = initialize()
sns.set_context("notebook")

data_dir = pathlib.Path("/mnt/data/torsk_experiments")
outdir = data_dir / "agulhas_3daymean_50x30"
pred_data_ncfiles = list(outdir.glob("pred_data_idx*.nc"))
valid_pred_length = 50
large_window = 200
small_window = 5
pred_plot_step = 10
prob_normality = 0.001
kuro_step = 3

pred_data_ncfiles, indices = sort_filenames(
    pred_data_ncfiles, return_indices=True)
params = Params(
    json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

figsize = (8, 6)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)

cmap = plt.get_cmap("inferno")
colors = cycle([cmap(i) for i in np.linspace(0, 1, 10)])
ax[0].set_ylabel("ESN Error")

imed_error, cycle_error, lstm_error = [], [], []
for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
    tqdm.write(pred_data_nc.as_posix())

    with nc.Dataset(pred_data_nc, "r") as src:

        pred_imed = src["imed"][:valid_pred_length]
        labels = src["labels"][:valid_pred_length]
        imed_error.append(pred_imed.mean(axis=0))

        if idx % pred_plot_step == 0:
            start = (idx+params.train_length)*kuro_step
            stop = start+kuro_step*valid_pred_length
            x = np.arange(start, stop, kuro_step)
            ax[0].plot(x, pred_imed, color=next(colors))

imed_error = np.array(imed_error)

imed_score, lw_mu, lw_std, sw_mu = sliding_score(
    imed_error, small_window=small_window, large_window=large_window)

indices = np.array(indices)
indices = (indices + params.train_length) * kuro_step
shifted_indices = indices + (valid_pred_length + small_window) * kuro_step

plot_start = indices[0]
plot_end = indices[-1]
plot_end += valid_pred_length*kuro_step
ax[-1].set_xlabel("Time [days]")

plot_indices = shifted_indices
ones = np.ones_like(plot_indices)

ax[1].plot([plot_start, plot_end], [prob_normality, prob_normality], ":",
    label=rf"$\Sigma={prob_normality}$", color="black")
ax[1].plot(plot_indices, imed_score, "-", label="ESN", color="C0")
ax[1].fill_between(plot_indices, ones, imed_score > prob_normality,
    label="Detected Anomaly", alpha=0.5, color="C0")
ax[1].set_ylim(1e-4, 1.1)
ax[1].set_ylabel("Normality")

bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 1}
for a, l in zip(ax, 'ABCD'):
    a.annotate(l, xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)

ax[1].set_yscale("log")
ax[1].legend(loc="lower left")
ax[2].set_yscale("log")
ax[2].legend(loc="lower left")
#ax[3].set_yscale("log")
#ax[3].legend(loc="lower left")
ax[0].annotate("Year 1", xy=(0.08, 1.02), xycoords="axes fraction")
ax[0].annotate("Year 2", xy=(0.27, 1.02), xycoords="axes fraction")
ax[0].annotate("Year 3", xy=(0.46, 1.02), xycoords="axes fraction")
ax[0].annotate("Year 4", xy=(0.67, 1.02), xycoords="axes fraction")
ax[0].annotate("Year 5", xy=(0.90, 1.02), xycoords="axes fraction")


for a in ax:
    a.set_xticks(np.arange(0, indices[-1], 365))
    a.set_xlim(plot_start, plot_end)
    a.grid(True)

finalize(args, fig, ax, loc="lower left")
