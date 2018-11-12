import pathlib

import requests
import numpy as np
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from torsk.data import normalize


_module_dir = pathlib.Path(__file__).absolute().parent
_data_dir = _module_dir / "../../data"


def _maybe_download(fname):
    if not fname.exists():
        dirname = fname.parent
        if not dirname.exists():
            dirname.mkdir(parents=True)
        print(f"Downloading SSH data to {fname} ...")
        kuro_url = "https://sid.erda.dk/share_redirect/d9pMpC1tUM"
        res = requests.get(kuro_url)
        with open(fname, "wb") as fi:
            fi.write(res.content)
        print("Done.")


class NetcdfDataset(Dataset):
    """Loads sea surface height (SSH) from a netCDF file of shape (seq, ydim,
    xdim) and returns chunks of of seq_length of the data.  The created
    inputs/labels sequences are shifted by one timestep so that they can be
    used to create a one-step-ahead predictor. Inputs/labels are normalized to
    (0, 1).

    Parameters
    ----------
    ncpath : pathlib.Path
        Path to the netCDF file
    seq_length : int
        length of the inputs/labels sequences 
    xslice : slice
        slice of the x-dimension to read from the SSH variable
    yslice : slice
        slice of the y-dimension to read from the SSH variable
    size : tuple
        (h, w) tuple that defines the resampled size of the SHH frames

    Returns
    -------
    inputs : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    labels : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    """
    def __init__(self, ncpath, seq_length, xslice, yslice, size):
        # TODO: solve downloading differently...
        _maybe_download(ncpath)
        self.seq_length = seq_length
        self.ncfile = nc.Dataset(ncpath, "r")
        self.data = self.ncfile["SSH"]
        self.nr_sequences = self.data.shape[0] - self.seq_length
        self.xslice = xslice
        self.yslice = yslice
        self.size = size

        if self.data.shape[0] <= seq_length:
            raise ValueError("First dimension of 'SSH' variable must be "
                    "larger than seq_length.")

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('MackeyDataset index out of range.')
        seq = self.data[
            index:index + self.seq_length + 1, self.yslice, self.xslice]
        seq = normalize(seq)
        ssh, mask = seq.data, seq.mask
        ssh[mask] = 0.

        seq = [tvf.to_pil_image(img[:, :, np.newaxis]) for img in seq]
        seq = [tvf.resize(img, self.size) for img in seq]
        seq = torch.cat([tvf.to_tensor(img) for img in seq], dim=0)
        seq = seq.reshape([self.seq_length + 1, -1])
        inputs, labels = seq[:-1], seq[1:]
        return torch.Tensor(inputs), torch.Tensor(labels)

    def __len__(self):
        return self.nr_sequences


def _read_kuro(xslice, yslice, demask=True):
    fname = _data_dir / "/ocean/kuro_SSH_3daymean.nc"
    _maybe_download(fname)
    with nc.Dataset(fname, "r") as df:
        array = df["SSH"][:, yslice, xslice]
        time = df["time"][:]
        if demask:
            ssh = array.data
            mask = array.mask
            ssh[mask] = 0.
        else:
            array.mask = np.logical_or(array.mask, array.data == -1)
            ssh = array
    return time, ssh
