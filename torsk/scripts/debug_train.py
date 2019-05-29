import pathlib

import click
import netCDF4 as nc

import torsk

@click.command("debug-train",
    short_help="Plot internal state contributions of different input maps")
@click.argument("train_data_nc", type=pathlib.Path)
@click.option("--train-step", "-i", type=int, default=200,
    help="Training state index to plot")
def cli(train_data_nc, train_step):

    with nc.Dataset(train_data_nc, "r") as src:
        inputs = src["inputs"][:]

    model = torsk.load_model(train_data_nc.parent, prefix="idx0-")
    model.params.debug = True
    model.forward(inputs)
