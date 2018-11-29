import numpy as np
import torch
from torch.utils.data import Dataset
from torsk.data.utils import normalize, dct2


def gauss2d(center, sigma, size, borders=[[-2, 2], [-2, 2]]):
    yc, xc = center
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])
    yy, xx = np.meshgrid(yy, xx)
    gauss = ((xx - xc)**2 + (yy - yc)**2) / (2 * sigma)**2
    return np.exp(-gauss)


class CircleDataset(Dataset):
    def __init__(self, train_length, pred_length, center, sigma, size):

        self.train_length = train_length
        self.pred_length = pred_length
        self.nr_sequences = center.shape[0] - train_length - pred_length

        self.seq = np.array(
            [gauss2d(c, sigma, size) for c in center])
        self.seq = normalize(self.seq)
        self.seq = self.seq.reshape((-1, size[0] * size[1]))

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('MackeyDataset index out of range.')
        train_end = index + self.train_length + 1
        train_seq = self.seq[index:train_end]
        inputs = torch.Tensor(train_seq[:-1])
        labels = torch.Tensor(train_seq[1:])
        pred_labels = torch.Tensor(
            self.seq[train_end:train_end + self.pred_length])
        return inputs, labels, pred_labels, torch.Tensor([[0]])

    def __len__(self):
        return self.nr_sequences


class DCTCircleDataset(Dataset):
    def __init__(self, train_length, pred_length, center, sigma, size, resize):

        self.train_length = train_length
        self.pred_length = pred_length
        self.nr_sequences = center.shape[0] - train_length - pred_length

        self.seq = np.array(
            [gauss2d(c, sigma, size) for c in center])

        self.seq = normalize(self.seq)
        self.dct = dct2(self.seq, resize)
        self.dct = self.dct.reshape((-1, resize[0] * resize[1]))

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('MackeyDataset index out of range.')
        train_end = index + self.train_length + 1
        train_seq = self.dct[index:train_end]
        inputs = torch.Tensor(train_seq[:-1])
        labels = torch.Tensor(train_seq[1:])
        pred_labels = torch.Tensor(
            self.dct[train_end:train_end + self.pred_length])
        return inputs, labels, pred_labels, torch.Tensor(self.seq[-self.pred_length:])

    def __len__(self):
        return self.nr_sequences


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    center = (1, 6)
    sigma = 0.5
    size = [100, 100]

    x = np.sin(np.arange(0, 2 * np.pi, 0.1))
    y = np.cos(0.5 * np.arange(0, 2 * np.pi, 0.1))
    center = [(i, j) for i, j in zip(x, y)]

    gauss = [gauss2d(c, sigma, size) for c in center]

    from torsk.visualize import animate_imshow
    anim = animate_imshow(gauss)
    plt.show()
