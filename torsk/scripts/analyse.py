import sys
import pathlib

import click
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.visualize import animate_double_imshow, write_video
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


def imed_plot(imed, predictions, labels):
    trivial_pred = np.tile(labels[0], [labels.shape[0], 1, 1])
    trivial_imed = imed_metric(labels, trivial_pred)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("IMED")
    ax.plot(imed, label="ESN")
    ax.plot(trivial_imed, label="trivial pred")
    ax.plot(trivial_imed, label="seasonal pred")
    ax.set_yscale("log")
    ax.legend()

    return fig, ax


@click.command("analyse", short_help="Plot IMED and create animation")
@click.argument("ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--save", is_flag=True, default=False,
    help="saves created plots/video at ncfile.{pdf/mp4}")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
def cli(ncfiles, save, show):

    sns.set_style("whitegrid")
    
    for ncfile in ncfiles:

        with nc.Dataset(ncfile, "r") as src:

            imed = src["imed"][:]
            click.echo(f"{ncfile.name}: IMED at step 100: {imed[100]}")

            predictions = src["outputs"][:]
            labels = src["labels"][:]

            fig, ax = imed_plot(imed, predictions, labels)

            if save:
                plt.savefig(ncfile.with_suffix(".pdf"))

                frames = np.concatenate([labels, predictions], axis=1)
                videofile = ncfile.with_suffix(".mp4").as_posix()
                write_video(videofile, frames)

            if show:
                plt.show() # show IMED plot

                anim = animate_double_imshow(labels, predictions)
                plt.show()
            else:
                plt.close()
