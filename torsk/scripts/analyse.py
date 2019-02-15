import sys
import pathlib

import click
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.visualize import animate_double_imshow, write_video
from torsk.imed import imed_metric
from torsk.data import detrend


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


def trivial_imed(labels):
    trivial_pred = np.tile(labels[0], [labels.shape[0], 1, 1])
    trivial_imed = imed_metric(labels, trivial_pred)
    return trivial_imed


def imed_plot(esn_imed, cycle_imed, labels):
    mean_imed = esn_imed.mean(axis=0)
    std_imed = esn_imed.std(axis=0)

    mean_cycle_imed = cycle_imed.mean(axis=0)
    std_cycle_imed = cycle_imed.std(axis=0)

    trivial_imeds = np.array([trivial_imed(l) for l in labels])
    mean_trivial_imed = trivial_imeds.mean(axis=0)
    std_trivial_imed = trivial_imeds.std(axis=0)

    fig, ax = plt.subplots(1, 1)
    x = np.arange(mean_imed.shape[0])
    ax.set_title("IMED")

    ax.plot(mean_cycle_imed, label="cycle pred")
    ax.fill_between(
        x, 
        mean_cycle_imed+std_cycle_imed,
        mean_cycle_imed-std_cycle_imed, alpha=0.5)

    ax.plot(mean_trivial_imed, label="trivial pred")
    ax.fill_between(
        x, 
        mean_trivial_imed+std_trivial_imed,
        mean_trivial_imed-std_trivial_imed, alpha=0.5)

    ax.plot(x, mean_imed, label="ESN")
    ax.fill_between(x, mean_imed+std_imed, mean_imed-std_imed, alpha=0.5)
    ax.legend()
    ax.set_yscale("log")

    return fig, ax


@click.command("analyse", short_help="Plot IMED and create animation")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--save", is_flag=True, default=False,
    help="saves created plots/video at pred_data_nc.{pdf/mp4}")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--cycle-length", "-c", default=None, type=int,
    help="manually set cycle length for trend/cycle based prediction."
         "If not set, this defaults to the value found in train_data_{...}.nc")
def cli(pred_data_ncfiles, save, show, cycle_length):
    
    sns.set_style("whitegrid")

    labels = []
    esn_imed = []
    cycle_imed = []
    
    # read preds/labels and create videos
    for ii, pred_data_nc in enumerate(pred_data_ncfiles):
        assert "pred_data" in pred_data_nc.as_posix()

        with nc.Dataset(pred_data_nc, "r") as src:

            esn_imed.append(src["imed"][:])
            click.echo(f"{pred_data_nc.name}: IMED at step 100: {esn_imed[ii][100]}")

            labels.append(src["labels"][:])
            prediction = src["outputs"]

            if save:
                click.echo("FIXME: write_video does not work")
                # frames = np.concatenate([labels[ii], prediction], axis=1)
                # videofile = pred_data_nc.with_suffix(".mp4").as_posix()
                # write_video(videofile, frames)

            if show:
                anim = animate_double_imshow(labels[ii], prediction, title="ESN Pred.")
                plt.show()

        train_data_nc = pred_data_nc.as_posix().replace("pred_data", "train_data")
        with nc.Dataset(train_data_nc, "r") as src:
            training_Ftxx = src["labels"][:]
            if cycle_length is None:
                cycle_length = src.cycle_length
            pred_length = labels[-1].shape[0]

            cpred, _ = detrend.predict_from_trend_unscaled(
                training_Ftxx, cycle_length, pred_length)

            cycle_imed.append(imed_metric(cpred, labels[-1]))

            if show:
                anim = animate_double_imshow(labels[ii], cpred, title="Cycle Pred.")
                plt.show()

    labels = np.array(labels)
    esn_imed = np.array(esn_imed)
    cycle_imed = np.array(cycle_imed)

    fig, ax = imed_plot(esn_imed, cycle_imed, labels)
    if save:
        directory = pred_data_nc.parent
        fname = directory.name
        directory = directory.as_posix()
        plt.savefig(f"{directory}/{fname}.pdf")
    if show:
        plt.show() # show IMED plot
    else:
        plt.close()
