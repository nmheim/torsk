import pathlib
import click


@click.command("detect", short_help="Find anomalies in multiple predicition runs")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--outfile", "-o", type=pathlib.Path, default=None, help="saves created plot")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
@click.option("--large-window", "-l", type=int, default=100,
    help="Large normality score window")
@click.option("--small-window", "-s", type=int, default=10,
    help="Small normality score window")
@click.option("--pred-plot-step", "-n", type=int, default=10,
    help="Only plot prediciton error every n steps")
@click.option("--prob-normality", type=float, default=0.05,
    help="Probability of normality")
@click.option("--mackey", is_flag=True, default=False,
    help="Set this flag to plot an additional mackey-verification plot")
def cli(pred_data_ncfiles, outfile, show, valid_pred_length,
        large_window, small_window, pred_plot_step, prob_normality, mackey):
    from itertools import cycle
    from tqdm import tqdm
    import numpy as np
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    import seaborn as sns

    from torsk.scripts.pred_perf import sort_filenames
    from torsk.data.utils import mackey_anomaly_sequence, normalize
    from torsk.imed import imed_metric
    from torsk.anomaly import sliding_score
    from torsk import Params

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    kuro = False
    if kuro:
        kuro_start = 0
        kuro_step = 5

    if mackey:
        ax_offset = 1
    else:
        ax_offset = 0

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")

    nr_plots = 4 if mackey else 3
    figsize = (8, 6) if mackey else (8, 5)
    fig, ax = plt.subplots(nr_plots, 1, sharex=True, figsize=figsize)
    ax[ax_offset].set_ylabel("Error")

    cmap = plt.get_cmap("inferno")
    colors = cycle([cmap(i) for i in np.linspace(0, 1, 10)])

    imed_error, cycle_error = [], []
    for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:

            pred_imed = src["imed"][:valid_pred_length]
            labels = src["labels"][:valid_pred_length]
            imed_error.append(pred_imed.mean(axis=0))

            if idx % pred_plot_step == 0:
                if kuro:
                    start = (idx+params.train_length)*kuro_step+kuro_start
                    stop = start+kuro_step*valid_pred_length
                    x = np.arange(start, stop, kuro_step)
                else:
                    x = np.arange(idx, idx + valid_pred_length)
                ax[ax_offset].plot(x, pred_imed, color=next(colors))

        cycle_data_nc = pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
        cpred = np.load(cycle_data_nc)[:valid_pred_length]
        cycle_imed = imed_metric(cpred, labels)
        cycle_error.append(cycle_imed.mean(axis=0))

    imed_error = np.array(imed_error)
    cycle_error = np.array(cycle_error)

    imed_score, lw_mu, lw_std, sw_mu = sliding_score(
        imed_error, small_window=small_window, large_window=large_window)
    cycle_score, _, _, _ = sliding_score(
        cycle_error, small_window=small_window, large_window=large_window)

    indices = np.array(indices)
    if kuro:
        indices = (indices + params.train_length) * kuro_step + kuro_start
        shifted_indices = indices + valid_pred_length * kuro_step
    else:
        shifted_indices = indices + valid_pred_length

    if mackey:
        mackey_seq, anomaly = mackey_anomaly_sequence(
            N=indices[-1] + params.train_length,
            anomaly_start=params.anomaly_start,
            anomaly_step=params.anomaly_step)
        mackey_seq = normalize(mackey_seq)

        length = anomaly[params.train_length:].shape[0]
        ax[0].plot(mackey_seq[params.train_length:], label="x-Component", color="black")
        ax[0].fill_between(
            np.arange(length), np.zeros(length), anomaly[params.train_length:],
            color="grey", alpha=0.5, label="True Anomaly")
        ax[0].legend(loc="lower left")

    plot_start = indices[0]
    plot_end = indices[-1]
    if kuro:
        plot_end += valid_pred_length*kuro_step
        ax[-1].set_xlabel("Time [days]")
    else:
        plot_end += valid_pred_length

    plot_indices = shifted_indices
    ones = np.ones_like(plot_indices)

    ax[ax_offset+1].plot([plot_start, plot_end], [prob_normality, prob_normality], ":",
        label=rf"$\Sigma={prob_normality}$", color="black")
    ax[ax_offset+1].plot(plot_indices, imed_score, "-", label="ESN", color="C0")
    ax[ax_offset+1].fill_between(plot_indices, ones, imed_score > prob_normality,
        label="Detected Anomaly", alpha=0.5, color="C0")
    ax[ax_offset+1].set_ylim(1e-3, 1.)
    ax[ax_offset+1].set_ylabel("Normality")

    ax[ax_offset+2].plot([plot_start, plot_end], [prob_normality, prob_normality], ":",
        label=rf"$\Sigma={prob_normality}$", color="black")
    ax[ax_offset+2].plot(plot_indices, cycle_score, "-.", label="Cycle", color="C1")
    ax[ax_offset+2].fill_between(plot_indices, ones, cycle_score > prob_normality,
        label="Detected Anomaly", alpha=0.5, color="C1")
    ax[ax_offset+2].set_ylim(1e-3, 1.)
    ax[ax_offset+2].set_ylabel("Normality")

    # ax[ax_offset+2].plot(plot_indices, imed_error, label=r"error", color="black")
    # ax[ax_offset+2].plot(plot_indices, lw_mu, label=r"$\mu_m$")
    # ax[ax_offset+2].plot(plot_indices, lw_std, label=r"$\sigma_m$")
    # ax[ax_offset+2].plot(plot_indices, sw_mu, label=r"$\mu_n$")


    bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 1}
    for a, l in zip(ax, 'ABCD'):
        a.annotate(l, xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)

    ax[ax_offset+1].set_yscale("log")
    ax[ax_offset+1].legend(loc="lower left")
    ax[ax_offset+2].set_yscale("log")
    ax[ax_offset+2].legend(loc="lower left")

    for a in ax:
        if kuro:
            a.set_xticks(np.arange(0, indices[-1], 365))
        a.set_xlim(plot_start, plot_end)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
