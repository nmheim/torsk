import logging
import pathlib

import requests
import numpy as np
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from torsk.data.utils import dct2, split_train_label_pred


_module_dir = pathlib.Path(__file__).absolute().parent
_data_dir = _module_dir / "../../data"
logger = logging.getLogger(__file__)


def _maybe_download(fname):
    if not fname.exists():
        dirname = fname.parent
        if not dirname.exists():
            dirname.mkdir(parents=True)
        logging.info(f"Downloading SSH data to {fname} ...")
        kuro_url = "https://sid.erda.dk/share_redirect/d9pMpC1tUM"
        res = requests.get(kuro_url)
        with open(fname, "wb") as fi:
            fi.write(res.content)
        logging.info("Done.")


def resample(sequence, size):
    sequence = [tvf.to_pil_image(img[:, :, np.newaxis]) for img in sequence]
    sequence = [tvf.resize(img, size) for img in sequence]
    sequence = torch.cat([tvf.to_tensor(img) for img in sequence], dim=0)
    return sequence


def central_slice(sequence, size):
    if sequence.ndim != 3:
        raise ValueError(f"Input sequence should have 3 dimensions, but has {sequence.ndim}.")
    ydim, xdim = sequence.shape[1:]
    ydif, xdif = ydim - size[0], xdim - size[1]
    ydif_even, xdif_even = (ydif % 2) == 0, (xdif % 2) == 0
    ystart = ydif // 2 if ydif_even else (ydif + 1) // 2
    xstart = xdif // 2 if xdif_even else (xdif + 1) // 2

    yend = -ystart if ydif_even else -ystart + 1
    xend = -xstart if xdif_even else -xstart + 1

    yend = ydim if yend == 0 else yend
    xend = xdim if xend == 0 else xend

    sequence = sequence[:, ystart:yend, xstart:xend]

    assert size[0] == sequence.shape[1]
    assert size[1] == sequence.shape[2]
    return sequence


def central_pad(sequence, size):
    if sequence.ndim != 3:
        raise ValueError(f"Input sequence should have 3 dimensions, but has {sequence.ndim}.")
    ydim, xdim = sequence.shape[1:]
    ydif, xdif = size[0] - ydim, size[1] - xdim
    ydif_even, xdif_even = (ydif % 2) == 0, (xdif % 2) == 0

    ypad_start = ydif // 2 if ydif_even else (ydif + 1) // 2
    ypad_end = ypad_start if ydif_even else ypad_start - 1
    xpad_start = xdif // 2 if xdif_even else (xdif + 1) // 2
    xpad_end = xpad_start if xdif_even else xpad_start - 1

    # ypad, xpad = ydif // 2, xdif // 2
    padding = [[0, 0], [ypad_start, ypad_end], [xpad_start, xpad_end]]
    return np.pad(sequence, padding, 'constant')


class NetcdfDataset(Dataset):
    """Loads sea surface height (SSH) from a netCDF file of shape (seq, ydim,
    xdim) and returns chunks of of seq_length of the data.  The created
    inputs/labels sequences are shifted by one timestep so that they can be
    used to create a one-step-ahead predictor.

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
    def __init__(self, ncpath, train_length, pred_length, xslice, yslice, size=None):

        self.train_length = train_length
        self.pred_length = pred_length
        self.ncfile = nc.Dataset(ncpath, "r")
        self.data = self.ncfile["SSH"]
        self.xslice = xslice
        self.yslice = yslice
        self.size = size

        self.seq_length = train_length + pred_length
        self.nr_sequences = self.data.shape[0] - self.seq_length

        if self.data.shape[0] <= self.seq_length:
            raise ValueError("First dimension of 'SSH' variable must be "
                             "larger than seq_length.")

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('Dataset index out of range.')
        seq = self.data[
            index:index + self.seq_length + 1, self.yslice, self.xslice]

        ssh, mask = seq.data, seq.mask
        ssh[mask] = 0.

        if self.size is not None:
            seq = resample(seq, self.size)
        seq = seq.reshape([self.seq_length + 1, -1])

        inputs, labels, pred_labels = split_train_label_pred(
            seq, self.train_length, self.pred_length)

        return inputs, labels, pred_labels, torch.Tensor([[0]])

    def __len__(self):
        return self.nr_sequences


class DCTNetcdfDataset(Dataset):
    """Loads sea surface height (SSH) from a netCDF file of shape (seq, ydim,
    xdim) and returns chunks the spatial DCT of seq_length of the data.  The
    created inputs/labels sequences are shifted by one timestep so that they
    can be used to create a one-step-ahead predictor.

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
        (h, w) tuple that defines the centered rectangle of kept DCT
        coefficients

    Returns
    -------
    inputs : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    labels : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    """
    def __init__(self, ncpath, train_length, pred_length, xslice, yslice, size=None):

        self.train_length = train_length
        self.pred_length = pred_length
        self.ncfile = nc.Dataset(ncpath, "r")
        self.data = self.ncfile["SSH"]
        self.xslice = xslice
        self.yslice = yslice
        self.size = size

        self.seq_length = train_length + pred_length
        self.nr_sequences = self.data.shape[0] - self.seq_length

        if self.data.shape[0] <= self.seq_length:
            raise ValueError("First dimension of 'SSH' variable must be "
                             "larger than seq_length.")

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('Dataset index out of range.')
        seq = self.data[
            index:index + self.seq_length + 1, self.yslice, self.xslice]

        ssh, mask = seq.data, seq.mask
        ssh[mask] = 0.

        ssh = seq[-self.pred_length:].copy()
        seq = dct2(seq, self.size)

        seq = seq.reshape([self.seq_length + 1, -1])

        inputs, labels, pred_labels = split_train_label_pred(
            seq, self.train_length, self.pred_length)

        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        pred_labels = torch.Tensor(pred_labels)
        ssh = torch.Tensor(ssh)

        return inputs, labels, pred_labels, ssh

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

class SCTNetcdfDataset(Dataset):
    """Loads sea surface height (SSH) from a netCDF file of shape (seq, ydim,
    xdim) and returns chunks the spatial DCT of seq_length of the data.  The
    created inputs/labels sequences are shifted by one timestep so that they
    can be used to create a one-step-ahead predictor.

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
        (h, w) tuple that defines the centered rectangle of kept DCT
        coefficients

    Returns
    -------
    inputs : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    labels : torch.Tensor
        sequence of shape (seq_length, ydim, xdim)
    """
    def __init__(self, ncpath, train_length, pred_length):

        self.train_length = train_length
        self.pred_length = pred_length
        self.ncfile = nc.Dataset(ncpath, "r")
        self.data = self.ncfile["SSH_SCT"]
        self.full_mask = self.ncfile["full_mask"];
        self.edge_mask = self.ncfile["edge_mask"];
        self.time      = self.ncfile["time"];
        self.xdims     = (self.ncfile.dimensions["nlat"].size,self.ncfile.dimensions["nlon"].size);
        self.kdims     = (self.ncfile.dimensions["nk1"].size, self.ncfile.dimensions["nk2"].size);        

        (nlat,nlon) = self.xdims;
        (nk1,nk2)   = self.kdims;
        self.basis1 = sct_basis(nlat,nk1);
        self.basis2 = sct_basis(nlon,nk2);
        
        self.seq_length = train_length + pred_length
        self.nr_sequences = self.data.shape[0] - self.seq_length

        if self.data.shape[0] <= self.seq_length:
            raise ValueError("First dimension of 'SSH' variable must be "
                             "larger than seq_length.")

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('Dataset index out of range.')

        seq = self.data[index:index + self.seq_length + 1]
        seq = seq.reshape([self.seq_length + 1, -1])

        inputs, labels, pred_labels = split_train_label_pred(
            seq, self.train_length, self.pred_length)

        ssh = sct2(pred_labels,self.basis1,self.basis2)
        
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        pred_labels = torch.Tensor(pred_labels)
        ssh = torch.Tensor(ssh)

        return inputs, labels, pred_labels, ssh

    def __len__(self):
        return self.nr_sequences
