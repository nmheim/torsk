import sys
import pathlib

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.visualize import animate_double_imshow
from torsk import hpopt
from torsk.imed import imed_metric


def read_run_metrics(fname):
    with nc.Dataset(fname, 'r') as src:
        imed = src["imed"][:]
        eucd = src["eucd"][:]
    return imed, eucd


def read_all_metrics(directory):
    pred_data_nc = directory.glob("pred_data_*.nc")
    imeds, eucds = [], []
    for fname in pred_data_nc:
        imed, eucd = read_run_metrics(fname)
        imeds.append(imed)
        eucds.append(eucd)
    
    return np.array(imeds), np.array(eucds)


sns.set_style("whitegrid")


outdirs = [pathlib.Path(path) for path in sys.argv[1:]]
print(outdirs)

fig, ax = plt.subplots(1, 2, sharey=True)
for outdir in outdirs:
    imeds, eucds = read_all_metrics(outdir)

    mean = imeds.mean(axis=0)
    err = imeds.std(axis=0)
    # err = std / imeds.shape[0]
    x = np.arange(imeds.shape[1])
    ax[0].plot(x, mean, label=outdir.name)
    ax[0].fill_between(x, mean+err, mean-err, alpha=0.5)

    mean = eucds.mean(axis=0)
    err = eucds.std(axis=0)
    # err = std / imeds.shape[0]
    x = np.arange(imeds.shape[1])
    ax[1].plot(x, mean, label=outdir.name)
    ax[1].fill_between(x, mean+err, mean-err, alpha=0.5)

plt.show()
raise


# imed/ed plot
# video
# imed at step 100

# trivial prediction
# decomposed prediction

# hp_paths = hpopt.get_hpopt_dirs(".")
# 
# metrics = []
# for path in hp_paths:
#     with nc.Dataset(path.joinpath("pred_data.nc"), "r") as src:
#         outputs = src["outputs"][:]
#         labels = src["labels"][:]
#         # anim = animate_double_imshow(outputs, labels)
#         # plt.show()
#         m = src["metric"][:]
#         plt.plot(m)
#         metrics.append(m)
# plt.show()


fname = "output/pred_data_idx0.nc"
with nc.Dataset(fname, "r") as src:
    real_pixels = src["labels"][:]
    predicted_pixels = src["outputs"][:]

# fig, ax = plt.subplots(1, 2)
# im = ax[0].imshow((real_pixels > 0.9).sum(axis=0))
# plt.colorbar(im, ax=ax[0])
# im = ax[1].imshow((predicted_pixels > 0.9).sum(axis=0))
# plt.colorbar(im, ax=ax[1])
# plt.show()

plt.plot(imed_metric(real_pixels, predicted_pixels), label="imed")
error = np.abs(predicted_pixels - real_pixels)
plt.plot(error.sum(axis=-1).sum(axis=-1), label="abs_err")
plt.show()

error[0,0,0] = 1.
error[0,1,0] = 0.
anim = animate_double_imshow(real_pixels, predicted_pixels)
plt.show()

