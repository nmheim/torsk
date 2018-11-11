import numpy as np
import torch
from torch.utils.data import Dataset


def _simulate_mackey(b=None, N=3000):
    c   = 0.2
    d   = 0.01
    tau = 17
    n   = 10

    yinit = np.array([0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076,
        1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
        1.0739, 1.0759])

    if b is None:
        b = np.zeros(N) + 0.1

    y = np.zeros(N)
    y[:yinit.shape[0]] = yinit

    for i in range(tau, N-1):
        yi = y[i] - b[i]*y[i] + c*y[i-tau]/(1+y[i-tau]**n)
        y[i+1] = yi
    return y


class MackeyDataset(Dataset):
    """Simulates a Mackey-Glass system for simulation_steps and returns chunks
    of of seq_length of the simulated system.  The created inputs/labels
    sequences are shifted by one timestep so that they can be used to create a
    one-step-ahead predictor.

    Parameters
    ----------
    seq_length : int
        length of the inputs/labels sequences 
    simulation_steps : int
        number of Mackey-Glass simulation steps

    Returns
    -------
    inputs : torch.Tensor
        sequence of shape (seq_length, 1)
    labels : torch.Tensor
        sequence of shape (seq_length, 1)
    """
    def __init__(self, seq_length, simulation_steps):
        if simulation_steps <= seq_length:
            raise ValueError('simulation_steps must be larger than seq_length.')

        self.simulation_steps = simulation_steps
        self.seq_length = seq_length
        self.nr_sequences = self.simulation_steps - self.seq_length

        self.seq = _simulate_mackey(N=simulation_steps)
        self.seq = self.seq.reshape((-1, 1))

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('MackeyDataset index out of range.')
        seq = self.seq[index:index + self.seq_length + 1]
        inputs, labels = seq[:-1], seq[1:]
        return torch.Tensor(inputs), torch.Tensor(labels)

    def __len__(self):
        return self.nr_sequences
        

if __name__ == "__main__":
    
    ds = MackeyDataset(100, 2000)
    inputs, labels = ds[0]

    import matplotlib.pyplot as plt
    plt.plot(inputs)
    plt.plot(labels)
    plt.show()
