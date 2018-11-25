import pathlib
import numpy as np
import matplotlib.pyplot as plt
from torsk.visualize import animate_imshow, animate_double_imshow
from torsk.data import FFTNetcdfDataset, NetcdfDataset
from torsk.data.ocean import fft, ifft


def invert_fft(sequence, size):
    ypad = size[0] - sequence.shape[1]
    xpad = size[1] - sequence.shape[2]
    padding = [[0, 0], [ypad // 2, ypad // 2], [xpad // 2, xpad // 2]]
    sequence = np.pad(sequence, padding, 'constant')
    sequence = np.fft.ifftshift(sequence, axes=[1, 2])
    sequence = np.fft.ifft2(sequence)
    return sequence

def ocean_test():
    _moduledir = pathlib.Path(__file__).absolute().parent
    ncfile = _moduledir.joinpath("../../data/ocean/kuro_SSH_3daymean_scaled.nc")
    
    xstart, xdim = 200, 100
    ystart, ydim = 100, 100
    xslice = slice(xstart, xstart + xdim)
    yslice = slice(ystart, ystart + ydim)
    orig_size = (100, 100)
    small_size = (15, 15)
    
    # dataset = FFTNetcdfDataset(ncfile, 50, 50, xslice, yslice, size=None)
    dataset = NetcdfDataset(ncfile, 50, 50, xslice, yslice, size=orig_size)
    
    inputs, _, _ = dataset[0]
    inputs = inputs.reshape([-1, orig_size[0], orig_size[1]])

    inputs_fft = fft(inputs, small_size)
    inv_inputs_fft = ifft(inputs_fft, orig_size)
    return inputs, inputs_fft, inv_inputs_fft


def square_test():
    
    orig_size = (100, 100)
    small_size = (30, 30)

    inputs = np.ones(small_size)
    inputs = np.pad(inputs, [35, 35], 'constant')
    inputs = inputs.reshape(1, orig_size[0], orig_size[1])
    
    inputs_fft = fft(inputs, small_size)
    inv_inputs_fft = ifft(inputs_fft, orig_size)
    return inputs, inputs_fft, inv_inputs_fft


if __name__ == "__main__":

    inputs, inputs_fft, inv_inputs_fft = square_test()
    inputs, inputs_fft, inv_inputs_fft = ocean_test()

    anim = animate_double_imshow(inputs, inv_inputs_fft.real)
    plt.show()

    anim = animate_double_imshow(
        np.log10(np.abs(inputs_fft)), inv_inputs_fft.real)
    plt.show()
