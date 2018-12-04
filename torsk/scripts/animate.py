import pathlib
import netCDF4 as nc
import matplotlib.pyplot as plt
import click
from torsk.visualize import animate_imshow


@click.command("animate", short_help="Animate the SSH component of a .nc file")
@click.option("--variable", "-v", type=str, default="SSH")
@click.argument("ncfile", type=pathlib.Path)
def cli(ncfile, variable):
    with nc.Dataset(ncfile, "r") as src:
        data = src[variable][:]

    anim = animate_imshow(data)  # NOQA
    plt.show()
