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
from torsk.anomaly import sliding_score
from torsk import Params


@click.command("detect", short_help="Find anomalies in multiple predicition runs")
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
@click.option("--mackey", is_flag=True, default=False,
    help="Set this flag to plot an additional mackey-verification plot")
def cli(pred_data_ncfiles, outfile, show, valid_pred_length, large_window, small_window, mackey):

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    nr_plots = 4 if mackey else 3
    fig, ax = plt.subplots(nr_plots, 1, sharex=True)

    ax[0].set_title(r"IMED$(\mathbf{y}, \mathbf{d})$")
    ax[1].set_title("Mean IMED/EUCD")
    ax[2].set_title(f"Normality Score. LW:{large_window} SW:{small_window}")

    cmap = plt.get_cmap("inferno")
    colors = cycle([cmap(i) for i in np.linspace(0, 1, 10)])

    imed_error, eucd_error = [], []
    for pred_data_nc, idx, color in tqdm(
            zip(pred_data_ncfiles, indices, colors), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:
            
            pred_imed = src["imed"][:valid_pred_length]
            pred_eucd = src["eucd"][:valid_pred_length]
            x = np.arange(idx, idx + valid_pred_length)

            ax[0].plot(x, pred_imed, color=color) 

            imed_error.append(np.mean(pred_imed))
            eucd_error.append(np.mean(pred_eucd))

    imed_error = np.array(imed_error)
    eucd_error = np.array(eucd_error)

    imed_score = sliding_score(
        imed_error, small_window=small_window, large_window=large_window)
    eucd_score = sliding_score(
        eucd_error, small_window=small_window, large_window=large_window)

    ax[1].plot(indices, imed_error, label=r"$\overline{IMED}$")
    ax[1].plot(indices, eucd_error, label=r"$\overline{EUDC}$")
    ax[1].legend()

    ax[2].plot(indices[large_window:], imed_score)
    ax[2].plot(indices[large_window:], eucd_score)
    ax[2].set_yscale("log")

    if mackey:
        mackey_seq, anomaly = mackey_anomaly_sequence(N=indices[-1]+params.train_length,
            anomaly_start=params.anomaly_start, anomaly_step=params.anomaly_step)
        mackey_seq = normalize(mackey_seq)

        ax[3].plot(mackey_seq[params.train_length:], label="x-component")
        ax[3].plot(anomaly[params.train_length:], label="anomaly label")
        ax[3].legend()

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
