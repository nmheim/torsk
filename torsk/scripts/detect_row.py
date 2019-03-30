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

    sns.set_style("whitegrid")
    sns.set_context("paper")

    pred_data_ncfiles, indices = sort_filenames(
        pred_data_ncfiles, return_indices=True)

    esn_error, cycle_error, all_labels, all_preds = [], [], [], []
    for pred_data_nc, idx in tqdm(zip(pred_data_ncfiles, indices), total=len(indices)):
        tqdm.write(pred_data_nc.as_posix())

        with nc.Dataset(pred_data_nc, "r") as src:

            labels = src["labels"][:valid_pred_length, row]
            outputs = src["outputs"][:valid_pred_length, row]
            pred_esn = np.abs(outputs - labels)

            esn_error.append(pred_esn[-1])
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

    im = ax[0].imshow(all_labels.T, aspect="auto")
    plt.colorbar(im, ax=ax[0])
    ax[0].set_ylabel("Truth")
    im = ax[1].imshow(all_preds.T, aspect="auto")
    plt.colorbar(im, ax=ax[1])
    ax[1].set_ylabel("Prediction")
    im = ax[2].imshow(esn_error.T, aspect="auto")
    plt.colorbar(im, ax=ax[2])
    ax[2].set_ylabel("ESN Error")
    im = ax[3].imshow(cycle_error.T, aspect="auto")
    plt.colorbar(im, ax=ax[3])
    ax[3].set_ylabel("Cycle Error")

    bbox = {"boxstyle": "round", "pad": 0.3, "fc": "white", "ec": "gray", "lw": 2}
    ax[0].annotate('A', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[1].annotate('B', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[2].annotate('C', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)
    ax[3].annotate('D', xy=(0.05, 0.8), xycoords='axes fraction', bbox=bbox)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()
