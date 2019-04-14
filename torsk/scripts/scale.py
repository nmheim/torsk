import pathlib
import click


@click.command("scale", short_help="Create a MinMaxScaler for a given .nc file")
@click.argument("ncfile", type=pathlib.Path)
@click.option("--outfile", "-o", type=pathlib.Path, required=True,
    help="Path to save scaled variable")
@click.option("--vmin", type=int, default=-1)
@click.option("--vmax", type=int, default=1)
def cli(ncfile, outfile, vmin, vmax):
    import json
    import netCDF4 as nc

    with nc.Dataset(ncfile, "r") as src:
        data = src["SSH"][:]
        dims = [(name, dim.size) for name, dim in src.dimensions.items()]

    dmin = data.min()
    dmax = data.max()
    scaled = (data - dmin) / (dmax - dmin)
    scaled = scaled * (vmax - vmin) + vmin

    with nc.Dataset(outfile, "w") as dst:
        for name, dim in dims:
            dst.createDimension(name, dim)
        dst.createVariable("SSH", data.dtype, [name for name, _ in dims])
        dst["SSH"][:] = scaled

    with open(str(outfile).replace(".nc", ".json"), "w") as dst:
        meta = {
            "original_minimum": float(dmin),
            "original_maximum": float(dmax)}
        json.dump(meta, dst)

    print(f"Saved at {outfile}")
