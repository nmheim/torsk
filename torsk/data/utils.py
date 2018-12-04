import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.fftpack import dct, idct

# Least-squares approximation to restricted DCT-III / Inverse DCT-II
def sct_basis(nx,nk):
    xs = np.arange(nx);
    ks = np.arange(nk);
    basis = 2*np.cos(np.pi*(xs[:,None]+0.5)*ks[None,:]/nx);
    return basis;

def isct(fx,basis):  
    fk,_,_,_ = sp.linalg.lstsq(basis,fx);
    return fk;

def sct(fk,basis):
    fx = np.dot(basis,fk)  
    return fx;

def isct2(Fxx,basis1, basis2):
    Fkx = isct(Fxx.T,basis2);
    Fkk = isct(Fkx.T,basis1);
    return Fkk

def sct2(Fkk,basis1, basis2):
    Fkx = sct(Fkk.T,basis1);
    Fxx = sct(Fkx.T,basis2);
    return Fxx


def _dct_2d(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')


def _idct_2d(coefficient):
    return idct(idct(coefficient.T, norm='ortho').T, norm='ortho')


def dct2(sequence, size):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    sequence : ndarray
        with shape (time, ydim, xdim)
    size : (ysize, xsize)
        determines how many DCT coefficents are kept

    Returns
    -------
    ndarray
        DCT coefficients with shape (time, ysize, xsize)
    """
    return np.array([_dct_2d(frame)[:size[0], :size[1]] for frame in sequence])


def idct2(sequence, size):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    sequence : ndarray
        with shape (time, ydim, xdim)
    size : (ysize, xsize)
        Pads the sequence with (nsize - ndim) zeros before calculating the
        inverse transform. Must be larger than sequence.shape[1:]

    Returns
    -------
    ndarray
        DCT coefficients with shape (time, ysize, xsize)
    """
    ydim, xdim = sequence.shape[1:]
    padding = [[0, size[0] - ydim], [0, size[1] - xdim]]
    iseq = [_idct_2d(np.pad(frame, padding, 'constant')) for frame in sequence]
    return np.array(iseq)


def normalize(data, vmin=None, vmax=None):
    """Normalizes data to values from 0 to 1.
    If vmin/vmax are given they are assumed to be the maximal
    values of data"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return (data - vmin) / np.abs(vmin - vmax)


def min_max_scale(data, vmin=0., vmax=1.):
    vrange = vmax - vmin
    dmin = data.min()
    drange = data.max() - dmin
    scale = vrange / drange
    shift = vmin - dmin * scale
    data *= scale
    data += shift
    return data


def _custom_collate(batch):
    """Transform batch such that inputs and labels have shape:

        (tot_seq_len, batch_size, nr_features)
    """
    def transpose(tensor):
        return torch.transpose(torch.stack(tensor), 0, 1)
    batch = [list(b) for b in zip(*batch)]
    batch = [transpose(b) for b in batch]
    return batch


class SeqDataLoader(DataLoader):
    """Custom Dataloader that defines a fixed custom collate function, so that
    the loader returns batches of shape (seq_len, batch, nr_features).
    """
    def __init__(self, dataset, **kwargs):
        if 'collate_fn' in kwargs:
            raise ValueError(
                'SeqDataLoader does not accept a custom collate_fn '
                'because it already implements one.')
        kwargs['collate_fn'] = _custom_collate
        super(SeqDataLoader, self).__init__(dataset, **kwargs)


def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels
