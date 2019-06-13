import pathlib
import click


@click.command("detect-row", short_help="Find anomalies in multiple predicition runs")
@click.argument("pred_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--outfile", "-o", type=pathlib.Path, default=None, help="saves created plot")
@click.option("--show/--no-show", is_flag=True, default=True,
    help="show plots/video or not")
@click.option("--valid-pred-length", "-p", type=int, default=50,
    help="Pick IMED at this index as value for error sequence")
@click.option("--row", "-r", type=int, required=True,
    help="Specifies row to pick from each frame")
def cli(pred_data_ncfiles, outfile, show, valid_pred_length, row):
    from tqdm import tqdm
    import numpy as np
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    import seaborn as sns

    from torsk.scripts.pred_perf import sort_filenames

    sns.set_context("paper")
    sns.set_style("whitegrid")

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)

    esn_error, cycle_error, all_labels, all_preds = [], [], [], []
    for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:

            labels = src["labels"][:valid_pred_length, row]
            outputs = src["outputs"][:valid_pred_length, row]
            pred_esn = np.abs(outputs - labels)

            esn_error.append(pred_esn.mean(axis=0))
            all_labels.append(labels[-1])
            all_preds.append(outputs[-1])

        cycle_data_nc = pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
        cpred = np.load(cycle_data_nc)[:valid_pred_length]
        pred_cycle = np.abs(cpred[:, row] - labels)
        cycle_error.append(pred_cycle.mean(axis=0))

    esn_error = np.array(esn_error)
    cycle_error = np.array(cycle_error)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)


    fig, ax = plt.subplots(4, 1, sharex=True)
    from torsk.params import Params
    params = Params(
        json_path=pred_data_ncfiles[0].parent / f"idx{indices[0]}-params.json")
    step = 5
    kuro_start = 0
    start = (indices[0] + params.train_length) * step + kuro_start
    stop = start + all_labels.shape[0] * step
    x = np.arange(start, stop, step)
    y = np.arange(all_labels.shape[1])
    x, y = np.meshgrid(x, y)

    im = ax[0].pcolormesh(x, y, all_labels.T, rasterized=True, linewidth=4, zorder=0)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_ylabel("Truth")
    im = ax[1].pcolormesh(x, y, all_preds.T, rasterized=True, linewidth=4, zorder=0)
    plt.colorbar(im, ax=ax[1])
    ax[1].set_ylabel("Prediction")
    im = ax[2].pcolormesh(x, y, esn_error.T, rasterized=True, linewidth=4, zorder=0)
    plt.colorbar(im, ax=ax[2])
    ax[2].set_ylabel("ESN Error")
    im = ax[3].pcolormesh(x, y, cycle_error.T, rasterized=True, linewidth=4, zorder=0)
    plt.colorbar(im, ax=ax[3])
    ax[3].set_ylabel("Cycle Error")

    bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 2}
    ax[0].annotate('A', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[1].annotate('B', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[2].annotate('C', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[3].annotate('D', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[0].annotate("Year 1", xy=(0.08, 1.05), xycoords="axes fraction")
    ax[0].annotate("Year 2", xy=(0.27, 1.05), xycoords="axes fraction")
    ax[0].annotate("Year 3", xy=(0.5, 1.05), xycoords="axes fraction")
    ax[0].annotate("Year 4", xy=(0.7, 1.05), xycoords="axes fraction")
    ax[0].annotate("Year 5", xy=(0.9, 1.05), xycoords="axes fraction")
    ax[3].set_xlabel("Time [days]")

    for a in ax:
        a.grid(True, axis="x", zorder=1000)
        a.set_xticks(np.arange(0, stop, 365))
        a.set_xlim(start, stop)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
