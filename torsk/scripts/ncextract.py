import click
import pathlib
from netCDF4 import Dataset
from tqdm import tqdm


def get_metadata(src, variable):
    ncattrs = {name: src[variable].getncattr(name)
               for name in src[variable].ncattrs()}
    dim_names = src[variable].dimensions
    dims = [(name, dim.size) for name, dim in src.dimensions.items() if name in dim_names]
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
    infiles = sorted(list(in_dir.glob(pattern)))

    with Dataset(infiles[0], "r") as src:
        ncattrs, dims, dtype = get_metadata(src, variable)

    with Dataset(outfile, "w") as dst:
        for name, dim in dims:
            if name == "time":
                dim = None
            dst.createDimension(name, dim)
        dst.createVariable(variable, dtype, [name for name, _ in dims])

        for ii, infile in tqdm(enumerate(infiles), desc="Extracting variables", total=len(infiles)):
            with Dataset(infile, "r") as src:
                data = src[variable][...]
            dst[variable][ii] = data[0]
