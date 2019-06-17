import pathlib
import torch
from torch.utils.data import DataLoader
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import torsk
from torsk.imed import imed_metric, metric_matrix
from torsk.models.torch_lstm import LSTM
from torsk.visualize import animate_double_imshow
from torsk.data.numpy_dataset import NumpyImageDataset

sns.set_style("whitegrid")

def esn_perf(outdir, animate=False):
    pred_data_nc_files = list(outdir.glob("pred_data_idx*.nc"))

    with nc.Dataset(pred_data_nc_files[0], "r") as src:
        img_shape = src["outputs"].shape[1:]
        pred_length = src["outputs"].shape[0]
        G = metric_matrix(img_shape)
    esn_error = np.empty(len(pred_data_nc_files), pred_length)
    trivial_error = np.empty(len(pred_data_nc_files), pred_length)


    G = None
    img_shape = None
    for i, pred_data_nc in tqdm(enumerate(pred_data_nc_files)):
        with nc.Dataset(pred_data_nc, 'r') as src:

            esn_error[i] = src["imed"][:]

            example_lbls = src["labels"][:]
            example_pred = src["outputs"][:]
            triv_pred = np.tile(example_lbls[0], (example_lbls.shape[0], 1, 1))
            trivial_error[i] = imed_metric(triv_pred, example_lbls, G=G)
    if animate:
        anim = animate_double_imshow(example_lbls, example_pred)
        plt.show()
    return np.array(error).mean(axis=0), np.array(trivial_error).mean(axis=0)


    
if __name__ == "__main__":
    from utils import initialize, finalize
    
    args = initialize()

    data_dir = pathlib.Path("/mnt/data/torsk_experiments")
    outdir = data_dir / "agulhas_3daymean_50x30"

    esn_error, trivial_error = esn_perf(outdir, animate=True)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(esn_error, label="ESN")
    ax.plot(trivial_error, label="trivial")
    ax.set_ylabel("Error")
    ax.set_xlabel("Days")
    finalize(args, fig, [ax], loc="upper left")
