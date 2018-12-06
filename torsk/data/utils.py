import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.fftpack import dct, idct, dctn, idctn
from scipy.linalg import lstsq
import scipy as sp


# Least-squares approximation to restricted DCT-III / Inverse DCT-II
def sct_basis(nx,nk):
    xs = np.arange(nx);
    ks = np.arange(nk);
    basis = 2*np.cos(np.pi*(xs[:,None]+0.5)*ks[None,:]/nx);        
    return basis;

def sct(fx,basis):  
    fk,_,_,_ = lstsq(basis,fx);
    return fk;

def isct(fk,basis):
    fx = np.dot(basis,fk);
    return fx;

def sct2(Fxx,basis1, basis2):
    Fkx = sct(Fxx.T,basis2);
    Fkk = sct(Fkx.T,basis1);
    return Fkk

def isct2(Fkk,basis1, basis2):
    Fkx = isct(Fkk.T,basis2);
    Fxx = isct(Fkx.T,basis1);
    return Fxx

def dct2(Fxx,nk1,nk2):
    Fkk = dctn(Fxx,norm='ortho')[:nk1,:nk2];
    return Fkk

def idct2(Fkk,nx1,nx2):
    Fxx = idctn(Fkk,norm='ortho',shape=(nx1,nx2));
    return Fxx

def idct2_sequence(Ftkk,xsize):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftkk : ndarray with shape (time, nk1, nk2)
    size : (ny,nx) determines the resolution of the image

    Returns
    -------
    Ftxx: ndarray with shape (time, ny,nx)
    """        
    Ftxx = idctn(Ftkk,norm='ortho',shape=xsize,axes=[1,2]);
    return Ftxx;


def dct2_sequence(Ftxx, ksize):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftxx : ndarray with shape (time, ydim, xdim)
    size : (nk1,nk2) determines how many DCT coefficents are kept

    Returns
    -------
    Ftkk: ndarray with shape (time, nk1, nk2)
    """    
    Ftkk = dctn(Ftxx,norm='ortho',axes=[1,2])[:,:ksize[0],:ksize[1]];
    return Ftkk;


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
