import pathlib
import netCDF4 as nc
import matplotlib.pyplot as plt
import click
from torsk.visualize import animate_imshow


@click.command("animate", short_help="Animate the SSH component of a .nc file")
@click.argument("ncfile", type=pathlib.Path)
def cli(ncfile):
    with nc.Dataset(ncfile, "r") as src:
        data = src["SSH"][:]

    anim = animate_imshow(data)
    plt.show()
