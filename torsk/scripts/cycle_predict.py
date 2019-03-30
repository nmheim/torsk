import pathlib
import click


@click.command("cycle-predict", short_help="Find anomalies in multiple predicition runs")
@click.argument("train_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--cycle-length", "-c", type=int, default=73,
    help="Cycle length for cycle-based prediction")
@click.option("--pred-length", "-p", type=int, default=100,
    help="Prediction length for cycle-based prediction")
@click.option("--train-length", "-t", type=int, default=730,
    help="Train length for cycle-based prediction. Must be mulitple of cycle!")
def cli(train_data_ncfiles, cycle_length, pred_length, train_length):
    """Create cycle-based prediction. Results are stored in
    `cycle_{train_data_nc.stem}.npy` and overwritten if already present.
    """
    from tqdm import tqdm
    import numpy as np
    import netCDF4 as nc

    from torsk.scripts.pred_perf import sort_filenames
    from torsk.data import detrend

    if train_length % cycle_length != 0:
        raise ValueError("Train length must be mulitple of cycle length.")

    train_data_ncfiles, indices = sort_filenames(train_data_ncfiles)

    for train_data_nc in tqdm(train_data_ncfiles):
        with nc.Dataset(train_data_nc, "r") as src:
            training_Ftxx = src["labels"][-train_length:]
            cpred, _ = detrend.predict_from_trend_unscaled(
                training_Ftxx, cycle_length=cycle_length, pred_length=pred_length)
            outfile = train_data_nc.parent / f"cycle_{train_data_nc.stem}.npy"
            tqdm.write(f"Saving cycle-based prediction at {outfile}")
            np.save(outfile, cpred)
