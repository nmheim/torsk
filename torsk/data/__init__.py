import numpy as np
import torch
from torch.utils.data import DataLoader


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

    inputs = batch[0]
    labels = batch[1]
    pred_labels = batch[2]
    return inputs, labels, pred_labels


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


from torsk.data.mackey import MackeyDataset
from torsk.data.ocean import NetcdfDataset
from torsk.data.sine import SineDataset

__all__ = ["MackeyDataset", "NetcdfDataset", "SineDataset"]
