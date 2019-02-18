import pathlib

import click
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.analyse import sort_filenames
from torsk.data.utils import mackey_anomaly_sequence


@click.command("detect", short_help="Find anomalies in multiple predicition runs")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--save", is_flag=True, default=False,
    help="saves created plots/video at pred_data_nc.{pdf/mp4}")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
def cli(pred_data_ncfiles, save, show):

    sns.set_style("whitegrid")
    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)

    valid_pred_length = 100
    fig, ax = plt.subplots(2, 1, sharex=True)

    for pred_data_nc, idx in zip(pred_data_ncfiles, indices):

        with nc.Dataset(pred_data_nc, "r") as src:
            
            pred_imed = src["imed"][:]
            x = np.arange(idx, idx + pred_imed.shape[0])
            ax[0].plot(x, pred_imed) 

    mackey, anomaly = mackey_anomaly_sequence()

    ax[1].plot(mackey[2000:])
    ax[1].plot(anomaly[2000:])

    plt.show()
