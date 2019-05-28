# coding: future_fstrings
import pathlib
import click


@click.command("anomaly-count", short_help="Create anomaly count map")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--outfile", "-o", type=pathlib.Path, default=None, help="saves created plot")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
@click.option("--large-window", "-l", type=int, default=20,
    help="Large normality score window")
@click.option("--small-window", "-s", type=int, default=3,
    help="Small normality score window")
@click.option("--normality-threshold", "-n", type=float, default=1e-2)
@click.option("--mask-file", type=pathlib.Path, default=None)
def cli(pred_data_ncfiles, outfile, show, valid_pred_length, large_window,
        small_window, normality_threshold, mask_file):
    from tqdm import tqdm
    import numpy as np
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    import seaborn as sns

    from torsk.scripts.pred_perf import sort_filenames
    from torsk.anomaly import sliding_score

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    pred_data_ncfiles, indices = sort_filenames(pred_data_ncfiles, return_indices=True)

    pixel_error, trivial_error, cycle_error = [], [], []
    for idx, pred_data_nc in tqdm(zip(indices, pred_data_ncfiles), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())
        with nc.Dataset(pred_data_nc, "r") as src:
            outputs = src["outputs"][:valid_pred_length]
            labels = src["labels"][:valid_pred_length]

        error_seq = np.abs(outputs - labels)
        error = np.mean(error_seq, axis=0)
        pixel_error.append(error)

        trivial_seq = np.abs(labels - labels[0])
        triv_err = np.mean(trivial_seq, axis=0)
        trivial_error.append(triv_err)

        cycle_data_nc = pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
        cpred = np.load(cycle_data_nc)[:valid_pred_length]
        error_seq = np.abs(cpred - labels)
        error = np.mean(error_seq, axis=0)
        cycle_error.append(error)

    pixel_error = np.array(pixel_error)
    trivial_error = np.array(trivial_error)
    cycle_error = np.array(cycle_error)

    pixel_score, _, _, _ = sliding_score(
        pixel_error, small_window=small_window, large_window=large_window)
    trivial_score, _, _, _ = sliding_score(
        trivial_error, small_window=small_window, large_window=large_window)
    cycle_score, _, _, _ = sliding_score(
        cycle_error, small_window=small_window, large_window=large_window)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    pixel_count = np.sum(pixel_score < normality_threshold, axis=0)
    trivial_count = np.sum(trivial_score < normality_threshold, axis=0)
    cycle_count = np.sum(cycle_score < normality_threshold, axis=0)

    if mask_file is not None:
        mask = np.load(mask_file)
        pixel_count = np.ma.masked_array(pixel_count, mask=mask)
        trivial_count = np.ma.masked_array(trivial_count, mask=mask)
        cycle_count = np.ma.masked_array(cycle_count, mask=mask)

    im = ax[0].imshow(trivial_count[::-1])
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    im = ax[1].imshow(cycle_count[::-1])
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    im = ax[2].imshow(pixel_count[::-1])
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 2}
    ax[0].annotate('A', xy=(0.05, 0.9), xycoords='axes fraction', bbox=bbox)
    ax[1].annotate('B', xy=(0.05, 0.9), xycoords='axes fraction', bbox=bbox)
    ax[2].annotate('C', xy=(0.05, 0.9), xycoords='axes fraction', bbox=bbox)

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
