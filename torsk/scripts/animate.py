import pathlib
import netCDF4 as nc
import matplotlib.pyplot as plt
import click
from torsk.visualize import animate_imshow, write_video


@click.command("animate", short_help="Animate the SSH component of a .nc file")
@click.argument("ncfile", type=pathlib.Path)
@click.option("--variable", "-v", type=str, default="SSH")
@click.option("--outfile", "-o", type=pathlib.Path, default=None)
@click.option("--show/--no-show", is_flag=True, default=True)
def cli(ncfile, variable, outfile, show):
    with nc.Dataset(ncfile, "r") as src:
        data = src[variable][:]

    if outfile is not None:
        if outfile.suffix == ".gif":
            anim = animate_imshow(data)  # NOQA
            anim.save(outfile.as_posix(), writer="imagemagick")
        else:
            write_video(outfile.as_posix(), data)

    if show:
        anim = animate_imshow(data)  # NOQA
        plt.show()
