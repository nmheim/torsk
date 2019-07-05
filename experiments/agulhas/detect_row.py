import pathlib
import click

from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.pred_perf import sort_filenames
from torsk.params import Params
sns.set_context("paper")
sns.set_style("whitegrid")

data_dir = pathlib.Path("/mnt/data/torsk_experiments")
outdir = data_dir / "agulhas_3daymean_50x30"
pred_data_ncfiles = list(outdir.glob("pred_data_idx*.nc"))
valid_pred_length = 25
use_col = True
row = 8
col = 41


pred_data_ncfiles, indices = sort_filenames(pred_data_ncfiles, return_indices=True)

esn_error, all_labels, all_preds = [], [], []
for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
    tqdm.write(pred_data_nc.as_posix())

    with nc.Dataset(pred_data_nc, "r") as src:

        if use_col:
            labels = src["labels"][:valid_pred_length, :, col]
            outputs = src["outputs"][:valid_pred_length, :, col]
        else:
            labels = src["labels"][:valid_pred_length, row]
            outputs = src["outputs"][:valid_pred_length, row]
        pred_esn = np.abs(outputs - labels)

        esn_error.append(pred_esn.mean(axis=0))
        all_labels.append(labels[-1])
        all_preds.append(outputs[-1])

esn_error = np.array(esn_error)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)


fig, ax = plt.subplots(3, 1, sharex=True)
params = Params(
    json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")
step = 1
kuro_start = 0
start = (indices[0] + params.train_length + valid_pred_length) * step + kuro_start
stop = start + all_labels.shape[0] * step
x = np.arange(start, stop, step)
y = np.arange(all_labels.shape[1])
x, y = np.meshgrid(x, y)

im = ax[0].pcolormesh(x, y, all_labels.T, rasterized=True, linewidth=4, zorder=0)
plt.colorbar(im, ax=ax[0])
ax[0].set_ylabel("Truth")
im = ax[1].pcolormesh(x, y, all_preds.T, rasterized=True, linewidth=4, zorder=0)
plt.colorbar(im, ax=ax[1])
ax[1].set_ylabel("Prediction")
im = ax[2].pcolormesh(x, y, esn_error.T, rasterized=True, linewidth=4, zorder=0)
plt.colorbar(im, ax=ax[2])
ax[2].set_ylabel("ESN Error")

bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 2}
ax[0].annotate('A', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
ax[1].annotate('B', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
ax[2].annotate('C', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
ax[2].set_xlabel("Time [frames]")

for a in ax:
    a.grid(True, axis="x", zorder=1000)
    a.set_xticks(np.arange(0, stop, 365))
    a.set_xlim(start, stop)

plt.tight_layout()
plt.show()
