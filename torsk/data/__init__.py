import torch
from torch.utils.data import DataLoader
from torsk.data.mackey import MackeyDataset


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
    return inputs, labels


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
