import pathlib

import click
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.scripts.analyse import sort_filenames
from torsk.data.utils import mackey_anomaly_sequence, normalize
from torsk import Params


@click.command("detect", short_help="Find anomalies in multiple predicition runs")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--save", is_flag=True, default=False,
    help="saves created plots/video at pred_data_nc.{pdf/mp4}")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
def cli(pred_data_ncfiles, save, show, valid_pred_length):

    sns.set_style("whitegrid")
    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    fig, ax = plt.subplots(3, 1, sharex=True)

    for pred_data_nc, idx in zip(pred_data_ncfiles, indices):
        print(pred_data_nc)

        with nc.Dataset(pred_data_nc, "r") as src:
            
            pred_imed = src["imed"][:valid_pred_length]
            x = np.arange(idx, idx + valid_pred_length)
            ax[0].plot(x, pred_imed) 

            ax[1].plot(idx+valid_pred_length, np.mean(pred_imed), "s", color="red")
            # ax[1].plot(idx+valid_pred_length, pred_imed[-1], "o", color="blue")
            ax[1].plot(idx+valid_pred_length, np.median(pred_imed), ".", color="black")

    mackey, anomaly = mackey_anomaly_sequence(
        anomaly_start=params.anomaly_start, anomaly_step=params.anomaly_step)
    mackey = normalize(mackey)

    ax[2].plot(mackey[params.train_length:])
    ax[2].plot(anomaly[params.train_length:])

    plt.show()
