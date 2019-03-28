import pathlib

import click
from tqdm import tqdm
import numpy as np
import netCDF4 as nc

from torsk.scripts.prediction_performance import sort_filenames
from torsk.data import detrend
from torsk import Params


@click.command("cycle-predict", short_help="Find anomalies in multiple predicition runs")
@click.argument("train_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--cycle-length", "-c", type=int, default=100,
    help="Cycle length for cycle-based prediction")
@click.option("--pred-length", "-p", type=int, default=100,
    help="Prediction length for cycle-based prediction")
def cli(train_data_ncfiles, cycle_length, pred_length):

    train_data_ncfiles, indices = sort_filenames(
        train_data_ncfiles, return_indices=True)
    params = Params(
        json_path=train_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    for train_data_nc, idx in tqdm(zip(train_data_ncfiles, indices), total=len(indices)):
        with nc.Dataset(train_data_nc, "r") as src:
            training_Ftxx = src["labels"][:]
            cpred, _ = detrend.predict_from_trend_unscaled(
                training_Ftxx, cycle_length=cycle_length, pred_length=pred_length)
            outfile = train_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
            tqdm.write(f"Saving cycle-based prediction at {outfile}")
            np.save(outfile, cpred)
