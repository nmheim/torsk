import pathlib
from itertools import cycle

import click
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.prediction_performance import sort_filenames
from torsk.data.utils import mackey_anomaly_sequence, normalize
from torsk.data import detrend
from torsk.anomaly import sliding_score
from torsk import Params


@click.command("detect-row", short_help="Find anomalies in multiple predicition runs")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--outfile", "-o",  type=pathlib.Path, default=None, help="saves created plot")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
@click.option("--large-window", "-l", type=int, default=100,
    help="Large normality score window")
@click.option("--small-window", "-s", type=int, default=10,
    help="Small normality score window")
@click.option("--row", "-r", type=int, required=True,
    help="Specifies row to pick from each frame")
def cli(
    pred_data_ncfiles, outfile, show, valid_pred_length,
    large_window, small_window, row):

    sns.set_style("whitegrid")
    sns.set_context("paper")

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    cmap = plt.get_cmap("inferno")
    colors = cycle([cmap(i) for i in np.linspace(0, 1, 10)])

    esn_error, trivial_error, all_labels, all_preds = [], [], [], []
    for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:
            
            labels = src["labels"][:valid_pred_length, row]
            label0 = np.tile(labels[0], (valid_pred_length, 1))
            outputs = src["outputs"][:valid_pred_length, row]
            pred_trivial = np.abs(labels - label0)
            pred_esn = np.abs(outputs - labels)

            esn_error.append(pred_esn.mean(axis=0)) # TODO: or just last frame instead of mean???
            trivial_error.append(pred_trivial.mean(axis=0))
            all_labels.append(labels[-1])
            all_preds.append(outputs[-1])

        # train_data_nc = pred_data_nc.parent / f"train_data_idx{idx}.nc"
        # with nc.Dataset(train_data_nc, "r") as src:
        #     training_Ftxx = src["labels"][:]
        #     cpred, _ = detrend.predict_from_trend_unscaled(
        #         training_Ftxx, cycle_length=cycle_length, pred_length=labels.shape[0])
        #     cycle_esn = esn_metric(cpred, labels)

        #     cycle_error.append(np.mean(cycle_esn))


    esn_error = np.array(esn_error)
    trivial_error = np.array(trivial_error)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # esn_score = sliding_score(
    #     esn_error, small_window=small_window, large_window=large_window)
    # esn_score = np.concatenate(
    #     [np.ones((large_window,) + esn_error.shape[1:]), esn_score])

    fig, ax = plt.subplots(4, 1, sharex=True)

    im = ax[0].imshow(all_labels.T, aspect="auto")
    plt.colorbar(im, ax=ax[0])
    ax[0].set_ylabel("Truth")
    im = ax[1].imshow(all_preds.T, aspect="auto")
    plt.colorbar(im, ax=ax[1])
    ax[1].set_ylabel("Prediction")
    im = ax[2].imshow(esn_error.T, aspect="auto")
    plt.colorbar(im, ax=ax[2])
    ax[2].set_ylabel("ESN Error")
    im = ax[3].imshow(trivial_error.T, aspect="auto")
    plt.colorbar(im, ax=ax[3])
    ax[3].set_ylabel("Trivial Error")
    # im = ax[4].imshow(np.log10(esn_score.T), aspect="auto")
    # plt.colorbar(im, ax=ax[4])

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
