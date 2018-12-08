import click
import pathlib
from netCDF4 import Dataset
from tqdm import tqdm


def get_dims(src, variable):
    dim_names = src[variable].dimensions
    dims = {(name, dim.size) for name, dim in src.dimensions.items()
            if name in dim_names}
    return dims


def get_metadata(src, variable):
    ncattrs = {name: src[variable].getncattr(name)
               for name in src[variable].ncattrs()}
    dims = get_dims(src, variable)
    dtype = src[variable].dtype
    return ncattrs, dims, dtype


def create_dims(dst, dims):
    for name, dim in dims:
        if not name in dst.dimensions:
            if name == "time":
                dim = None
            dst.createDimension(name, dim)


@click.command("ncextract",
    short_help="Extract a variable from a number of .nc files")
@click.argument("infiles", type=pathlib.Path, nargs=-1)
@click.option("--outfile", "-o", type=pathlib.Path, required=True)
@click.option("--variable", "-V", type=str, default="SSH")
def cli(infiles, outfile, variable):

    if len(infiles) == 0:
        text = click.get_text_stream("stdin").read()
        infiles = text.split("\n")

    if outfile.exists():
        raise ValueError(f"{outfile} already exists!")

    with Dataset(outfile, "w") as dst:

        with Dataset(infiles[0], "r") as src:
            for var in [variable, "TLAT", "TLONG"]:
                dims = get_dims(src, var)
                create_dims(dst, dims)

                dtype = src[var].dtype
                dst.createVariable(var, dtype, src[var].dimensions)

                ncattrs = {name: src[var].getncattr(name)
                           for name in src[var].ncattrs()}
                dst[var].setncatts(ncattrs)

            tlat, tlon = src["TLAT"][:], src["TLONG"][:]

        dst.createVariable("time", dtype, ["time"])

        dst["TLAT"][:] = tlat
        dst["TLONG"][:] = tlon

        desc = "Extracting variables"
        total = len(infiles)
        for ii, infile in tqdm(enumerate(infiles), desc=desc, total=total):
            with Dataset(infile, "r") as src:
                data = src[variable][...]
            dst[variable][ii] = data[0]
