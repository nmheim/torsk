import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc

data_dir = pathlib.Path.home() / "erda_ro/Ocean/esn"

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", default=False, action="store_true")
    parser.add_argument("--outfile", "-o", type=str, default=None)
    args = parser.parse_args()
    
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    return args


def finalize(args, fig, ax, loc=None, bbox_to_anchor=None, frameon=True):
    lgd = []
    if loc is not None or bbox_to_anchor is not None:
        for a in ax:
            handles, labels = a.get_legend_handles_labels()
            if handles:
                l = a.legend(frameon=frameon, loc=loc, bbox_to_anchor=bbox_to_anchor)
                lgd.append(l)
    plt.tight_layout()
    if args.outfile is not None:
        plt.savefig(
            args.outfile, transparent=True, bbox_inches="tight", bbox_extra_artists=lgd)
    if not args.no_show:
        plt.show()
    plt.close()


def read_imed(pred_data_ncfiles, only_first_n=100, read_cycle_pred=True, return_cycle_pred=False):
    import numpy as np
    from torsk.scripts.pred_perf import sort_filenames
    from torsk.imed import imed_metric
    from tqdm import tqdm
    metric_log_idx = 25
    labels = []
    esn_imed, cycle_imed = [], []
    pred_data_ncfiles, indices = sort_filenames(pred_data_ncfiles, return_indices=True)
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
    
        if read_cycle_pred:
            cycle_pred_file = pred_data_nc.parent / f"cycle_pred_data_idx{idx}.npy"
            if cycle_pred_file.exists():
                cpred = np.load(cycle_pred_file)[:labels[0].shape[0]]
                cycle_imed.append(imed_metric(cpred, labels[-1]))
            else:
                raise ValueError(
                    f"{cycle_pred_file} does not exist. "
                    "Cannot compute cycle prediction. "
                    "Create it with `torsk cycle-predict`")
    if return_cycle_pred:
        return np.array(esn_imed), np.array(cycle_imed), np.array(labels), cpred
    return np.array(esn_imed), np.array(cycle_imed), np.array(labels)

