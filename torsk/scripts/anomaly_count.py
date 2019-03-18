import pathlib
from itertools import cycle

import click
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.prediction_performance import sort_filenames
from torsk.anomaly import sliding_score
from torsk import Params


@click.command("anomaly-count", short_help="Create anomaly count map")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--outfile", "-o",  type=pathlib.Path, default=None, help="saves created plot")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
@click.option("--large-window", "-l", type=int, default=20,
    help="Large normality score window")
@click.option("--small-window", "-s", type=int, default=3,
    help="Small normality score window")
@click.option("--normality-threshold", "-n", type=float, default=1e-2)
def cli(pred_data_ncfiles, outfile, show, valid_pred_length, large_window, small_window, normality_threshold):

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    pred_data_ncfiles, indices = sort_filenames(
            pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    pixel_error, trivial_error = [], []
    for pred_data_nc in tqdm(pred_data_ncfiles, total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())
        with nc.Dataset(pred_data_nc, "r") as src:
            outputs = src["outputs"][:valid_pred_length]
            labels = src["labels"][:valid_pred_length]
            
            error_seq = np.abs(outputs - labels)
            error = np.mean(error_seq, axis=0)
            pixel_error.append(error)

            trivial_seq = np.abs(outputs - labels[0])
            triv_err = np.mean(trivial_seq, axis=0)
            trivial_error.append(triv_err)

    pixel_error = np.array(pixel_error)
    trivial_error = np.array(trivial_error)

    pixel_score = sliding_score(
        pixel_error, small_window=small_window, large_window=large_window)
    trivial_score = sliding_score(
        trivial_error, small_window=small_window, large_window=large_window)

    fig, ax = plt.subplots(1,2)
    pixel_count = np.sum(pixel_score < normality_threshold, axis=0)
    trivial_count = np.sum(trivial_score < normality_threshold, axis=0)
    im = ax[0].imshow(pixel_count[::-1])
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    im = ax[1].imshow(trivial_count[::-1])
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
