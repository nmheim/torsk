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
from torsk.imed import imed_metric
from torsk.anomaly import sliding_score
from torsk import Params


@click.command("detect", short_help="Find anomalies in multiple predicition runs")
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
@click.option("--pred-plot-step", "-n", type=int, default=10,
    help="Only plot prediciton error every n steps")
@click.option("--prob-normality", type=float, default=0.05,
    help="Probability of normality")
@click.option("--mackey", is_flag=True, default=False,
    help="Set this flag to plot an additional mackey-verification plot")
def cli(
    pred_data_ncfiles, outfile, show, valid_pred_length,
    large_window, small_window, pred_plot_step, prob_normality, mackey):

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    nr_plots = 4 if mackey else 3
    figsize = (8,6)
    fig, ax = plt.subplots(nr_plots, 1, sharex=True, figsize=figsize)

    # ax[0].set_title(r"IMED$(\mathbf{y}, \mathbf{d})$")
    # ax[1].set_title("Mean IMED/EUCD")
    # ax[2].set_title(f"Normality Score. LW:{large_window} SW:{small_window}")

    cmap = plt.get_cmap("inferno")
    colors = cycle([cmap(i) for i in np.linspace(0, 1, 10)])

    imed_error, cycle_error, trivial_error = [], [], []
    for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:
            
            pred_imed = src["imed"][:valid_pred_length]
            labels = src["labels"][:valid_pred_length]
            label0 = np.tile(labels[0], (valid_pred_length, 1, 1))
            outputs = src["outputs"][:valid_pred_length]
            pred_trivial = imed_metric(labels, label0)

            imed_error.append(pred_imed[-1])
            trivial_error.append(pred_trivial[-1])

            if  idx % pred_plot_step == 0:
                x = np.arange(idx, idx + valid_pred_length)
                ax[0].plot(x, pred_imed, color=next(colors)) 

        cpred = np.load(pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy")[:valid_pred_length]
        cycle_imed = imed_metric(cpred, labels)
        cycle_error.append(cycle_imed[-1])


    imed_error = np.array(imed_error)
    trivial_error = np.array(trivial_error)
    cycle_error = np.array(cycle_error)

    imed_score = sliding_score(
        imed_error, small_window=small_window, large_window=large_window)
    trivial_score = sliding_score(
        trivial_error, small_window=small_window, large_window=large_window)
    cycle_score = sliding_score(
        cycle_error, small_window=small_window, large_window=large_window)

    shifted_indices = np.array(indices) + valid_pred_length

    ax[1].plot(shifted_indices, imed_error, label="ESN")
    ax[1].plot(shifted_indices, trivial_error, label="trivial")
    ax[1].plot(shifted_indices, cycle_error, label="cycle")
    ax[1].legend(loc="lower left")

    ax[2].plot(shifted_indices[large_window:], imed_score, label="ESN")
    ax[2].plot(shifted_indices[large_window:], trivial_score, label="trivial")
    ax[2].plot(shifted_indices[large_window:], cycle_score, label="cycle")
    ax[2].plot(
        indices, np.zeros_like(indices)+prob_normality,
        label=rf"$\Sigma={prob_normality}$", color="black")
    ax[2].set_yscale("log")
    ax[2].legend(loc="lower left")

    ax[0].annotate('A', xy=(0.05, 0.8), xycoords='axes fraction',
        bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})
    ax[1].annotate('B', xy=(0.05, 0.8), xycoords='axes fraction',
        bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})
    ax[2].annotate('C', xy=(0.05, 0.8), xycoords='axes fraction',
        bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})


    if mackey:
        mackey_seq, anomaly = mackey_anomaly_sequence(N=indices[-1]+params.train_length,
            anomaly_start=params.anomaly_start, anomaly_step=params.anomaly_step)
        mackey_seq = normalize(mackey_seq)

        length = anomaly[params.train_length:].shape[0]
        ax[3].plot(mackey_seq[params.train_length:], label="x-component")
        ax[3].fill_between(
            np.arange(length), np.zeros(length), anomaly[params.train_length:],
            color="grey", alpha=0.5, label="Anomaly")
        ax[3].legend(loc="lower left")
        ax[3].annotate('D', xy=(0.05, 0.8), xycoords='axes fraction',
            bbox={"boxstyle":"round", "pad":0.3, "fc":"white", "ec":"gray", "lw":2})


    for a in ax: a.set_xlim(0, len(indices))
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
