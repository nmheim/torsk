# coding: future_fstrings
import pathlib

import click
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

from torsk.visualize import animate_triple_imshow, write_video
from torsk.imed import imed_metric


def trivial_imed(labels):
    trivial_pred = np.tile(labels[0], [labels.shape[0], 1, 1])
    trivial_imed = imed_metric(labels, trivial_pred)
    return trivial_imed


def imed_plot(esn_imed, cycle_imed, labels):
    N = esn_imed.shape[0]
    mean_imed = esn_imed.mean(axis=0)
    std_imed = esn_imed.std(axis=0) / N

    mean_cycle_imed = cycle_imed.mean(axis=0)
    std_cycle_imed = cycle_imed.std(axis=0) / N

    trivial_imeds = np.array([trivial_imed(l) for l in labels])
    mean_trivial_imed = trivial_imeds.mean(axis=0)
    std_trivial_imed = trivial_imeds.std(axis=0) / N

    fig, ax = plt.subplots(1, 1)
    x = np.arange(mean_imed.shape[0])

    ax.plot(x, mean_trivial_imed, ":", label="Trivial", color="C2")
    ax.fill_between(
        x,
        mean_trivial_imed + std_trivial_imed,
        mean_trivial_imed - std_trivial_imed, alpha=0.5, color="C2")

    ax.plot(x, mean_cycle_imed, "-.", label="Cycle-based", color="C1")
    ax.fill_between(
        x,
        mean_cycle_imed + std_cycle_imed,
        mean_cycle_imed - std_cycle_imed, alpha=0.5, color="C1")

    ax.plot(x, mean_imed, label="ESN", color="C0")
    ax.fill_between(x,
        mean_imed + std_imed,
        mean_imed - std_imed, alpha=0.5, color="C0")

    ax.set_ylabel(r"Error")
    ax.legend(loc="upper right")

    return fig, ax


def sort_filenames(files, return_indices=False):
    """Sorts filenames by a component in the path with keyword `idx` like this:

        pred_data_idx1.nc
        pred_data_idx100.nc
        pred_data_idx10.nc

    becomes:

        pred_data_idx1.nc
        pred_data_idx10.nc
        pred_data_idx100.nc
    """
    indices = [[s for s in f.stem.split("_") if "idx" in s][0] for f in files]
    indices = [int(idx.replace("idx", "")) for idx in indices]
    sorted_files = [f for _, f in sorted(zip(indices, files))]
    if return_indices:
        return sorted_files, sorted(indices)
    else:
        return sorted_files


@click.command("pred-perf", short_help="Plot IMED and create animation")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--save-video", default=None, type=click.Choice(["mp4", "gif"]),
    help="saves created video at pred_data_idx{idx}_nc.suffix")
@click.option("--outfile", "-o", type=pathlib.Path, default=None,
    help="save final plot in outfile path.")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plot/videos or not")
@click.option("--ylogscale", default=False, is_flag=True)
@click.option("--metric-log-idx", "-i", default=25, type=int,
    help="Prints metric (e.g. IMED) at given index.")
@click.option("--xlim", default=None, type=int)
@click.option("--plot-label", default=None, type=str)
@click.option("--only-first-n", "-n", type=int, default=None,
    help="Evaluate only first n files (for testing)")
@click.option("--sns-context", default="talk",
    type=click.Choice(["notebook", "paper", "talk", "poster"]))
def cli(pred_data_ncfiles, save_video, outfile, show, ylogscale,
        metric_log_idx, xlim, plot_label, only_first_n, sns_context):
    """Create animations and averaged performance plot of prediction files.
    The ESN `pred_data_ncfiles` must be named like: pred_data_idx0.nc
    The cycle prediction files are assumed to be in the same directory with
    names like: cycle_pred_data_idx0.npy as created with `torsk cycle-predict`
    """
    sns.set_style("whitegrid")
    sns.set_context(sns_context)

    labels = []
    esn_imed, cycle_imed = [], []
    pred_data_ncfiles, indices = sort_filenames(pred_data_ncfiles, return_indices=True)

    if only_first_n is not None:
        pred_data_ncfiles = pred_data_ncfiles[:only_first_n]
        indices = indices[:only_first_n]

    # read prediction files and animate
    for ii, (idx, pred_data_nc) in tqdm(
            enumerate(zip(indices, pred_data_ncfiles)), total=len(pred_data_ncfiles)):
        assert "pred_data" in pred_data_nc.as_posix()

        with nc.Dataset(pred_data_nc, "r") as src:

            esn_imed.append(src["imed"][:])
            tqdm.write(
                f"{pred_data_nc.name}: IMED at step {metric_log_idx}: "
                f"{esn_imed[ii][metric_log_idx]}")

            labels.append(src["labels"][:])
            outputs = src["outputs"][:]

        cycle_pred_file = pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
        if cycle_pred_file.exists():
            cpred = np.load(cycle_pred_file)[:labels[0].shape[0]]
            cycle_imed.append(imed_metric(cpred, labels[-1]))
        else:
            raise ValueError(
                f"{cycle_pred_file} does not exist. "
                "Cannot compute cycle prediction. "
                "Create it with `torsk cycle-predict`")

        # japan = np.load("/home/niklas/Downloads/japan.npy")
        # japan = np.tile(japan, [cpred.shape[0], 1, 1])
        # cpred = np.ma.masked_array(cpred, mask=japan)[:, ::-1]
        # labels[ii] = np.ma.masked_array(labels[ii], mask=japan)[:, ::-1]
        # outputs = np.ma.masked_array(outputs, mask=japan)[:, ::-1]

        if save_video is not None:
            frames = np.concatenate([labels[ii], outputs], axis=1)
            videofile = pred_data_nc.with_suffix(f".{save_video}").as_posix()
            if save_video == "gif":
                anim = animate_triple_imshow(labels[ii], outputs, cpred,
                    axes_labels=["Truth", "ESN", "Cycle", "Trivial"])
                anim.save(videofile, writer="imagemagick")
            else:
                write_video(videofile, frames)

        if show:
            anim = animate_triple_imshow(
                labels[ii], outputs, cpred,
                axes_labels=["Truth", "ESN", "Cycle", "Trivial"])
            plt.show()

    # plot performance
    labels = np.array(labels)
    esn_imed, cycle_imed = np.array(esn_imed), np.array(cycle_imed)

    fig, ax = imed_plot(esn_imed, cycle_imed, labels)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if plot_label is not None:
        bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 2}
        ax.annotate(plot_label, xy=(0.05, 0.9), xycoords='axes fraction', bbox=bbox)
    plt.tight_layout()
    if ylogscale:
        ax.set_yscale("log")
    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
