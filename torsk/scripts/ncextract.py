import click
import pathlib
from netCDF4 import Dataset
from tqdm import tqdm


def get_metadata(src):
    ncattrs = {name: src[variable].getncattr(name)
               for name src[variable].ncattrs()}
    dims = [(name, dim.size) for name, dim in src.dimensions.items()]
    dtype = src[variable].dtype
    return ncattrs, dims, dtype


@click.command("ncextract",
    short_help="Extract a variable from a number of .nc files")
@click.option("--in-dir", "-i", type=pathlib.Path, required=True)
@click.option("--outfile", "-o", type=pathlib.Path, required=True)
@click.option("--pattern", "-p", type=str, default="*.nc")
@click.option("--variable", "-V", type=str, default="SSH")
def cli(in_dir, outfile, pattern, variable):

    if outfile.exists():
        raise ValueError(f"{outfile} already exists!")

    print("Matching files ...")
    infiles = list(in_dir.glob(pattern))

    with Dataset(infiles[0], "r") as src:
        ncattrs, dims, dtype get_metadata(src)

    with Dataset(outfile, "w") as dst:
        for name, dim in dims:
            dst.createDimension(name, dim)
        dst.createVariable(variable, dtype, [name for name, _ in dims])

        for ii, infile in tqdm(enumerate(infiles), desc="Extracting variables"):
            with Dataset(infile, "r") as src:
                data = src[variable][...]
            dst[variable][ii] = data
