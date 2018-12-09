import numpy as np
import torch
from torch.utils.data import Dataset
from torsk.data.utils import normalize, dct2_sequence, idct2_sequence, split_train_label_pred

def gauss2d(center, sigma, size, borders=[[-2, 2], [-2, 2]]):
    yc, xc = center
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])
    yy, xx = np.meshgrid(yy, xx)
    gauss = ((xx - xc)**2 + (yy - yc)**2) / (2 * sigma)**2
    return np.exp(-gauss)


class CircleDataset(Dataset):
    def __init__(self, train_length, pred_length, center, sigma, xsize,ksize=None,domain="pixels"):

        self.train_length = train_length
        self.pred_length = pred_length
        self.nr_sequences = center.shape[0] - train_length - pred_length

        seq = np.array([gauss2d(c, sigma, xsize) for c in center])
        seq = normalize(seq)

        if(domain=="DCT"):
            if(ksize==None):
                ksize = xsize;
            seq = dct2_sequence(seq,ksize);
            seq = seq.reshape((-1, ksize[0] * ksize[1]))            
        else:
            seq = seq.reshape((-1, xsize[0] * xsize[1]))

        print("seq:",seq.shape)
        self.seq = seq;

        self.xsize  = xsize;
        self.ksize  = ksize;
        self.domain = domain;
        
    def to_image(self,data):
        seq = data.numpy();
        if(self.domain=="DCT"):
            seq = seq.reshape((-1,self.ksize[0],self.ksize[1]));
            return idct2_sequence(seq,self.xsize);
        else:
            return seq.reshape((-1,self.xsize[0],self.xsize[1]));
        
    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('CircleDataset index out of range.')
        sub_seq = self.seq[index:index + self.train_length + self.pred_length + 1]
        inputs, labels, pred_labels = split_train_label_pred(
            sub_seq, self.train_length, self.pred_length)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        pred_labels = torch.Tensor(pred_labels)
        return inputs, labels, pred_labels, torch.Tensor([[0]])

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
