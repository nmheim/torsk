import pathlib
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.pred_perf import sort_filenames
from torsk.anomaly import sliding_score
from utils import initialize, finalize

args = initialize()

data_dir = pathlib.Path("/mnt/data/torsk_experiments")
outdir = data_dir / "agulhas_3daymean_50x30"
pred_data_ncfiles = list(outdir.glob("pred_data_idx*.nc"))
valid_pred_length = 10
large_window = 100
small_window = 5
normality_threshold = 0.001

sns.set_context("notebook")

pred_data_ncfiles, indices = sort_filenames(pred_data_ncfiles, return_indices=True)

pixel_error, trivial_error, cycle_error = [], [], []
for idx, pred_data_nc in tqdm(zip(indices, pred_data_ncfiles), total=len(indices)):
    tqdm.write(pred_data_nc.as_posix())
    with nc.Dataset(pred_data_nc, "r") as src:
        outputs = src["outputs"][:valid_pred_length]
        labels = src["labels"][:valid_pred_length]

    error_seq = np.abs(outputs - labels)
    error = np.mean(error_seq, axis=0)
    pixel_error.append(error)

pixel_error = np.array(pixel_error)

pixel_score, _, _, _ = sliding_score(
    pixel_error, small_window=small_window, large_window=large_window)

fig, ax = plt.subplots(1, 1)
pixel_count = np.sum(pixel_score < normality_threshold, axis=0)

im = ax.imshow(pixel_count[::-1])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

finalize(args, fig, [ax], loc="lower left")
