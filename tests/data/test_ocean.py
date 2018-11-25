import netCDF4 as nc
import numpy as np
from torsk.data.ocean import NetcdfDataset
from torsk.data.ocean import central_slice, central_pad


def test_nc_dataset(tmpdir):

    ncpath = tmpdir.join('test.nc')
    with nc.Dataset(ncpath, 'w') as ncfile:
        ncfile.createDimension('time', 100)
        ncfile.createDimension('ydim', 10)
        ncfile.createDimension('xdim', 10)
        ssh = ncfile.createVariable('SSH', np.float32, ['time', 'ydim', 'xdim'])
        ssh[:] = np.random.uniform(size=[100, 10, 10])

    dataset = NetcdfDataset(
        ncpath, train_length=30, pred_length=20,
        xslice=slice(0, 10), yslice=slice(0, 10),
        size=[10, 10])

    inputs, labels, pred_labels = dataset[0]

    assert inputs.shape == (30, 100)
    assert labels.shape == (30, 100)
    assert pred_labels.shape == (20, 100)
    assert np.all(inputs[1].numpy() == labels[0].numpy())


def test_central_slice():

    imgs = np.arange(10)
    imgs = np.tile(imgs, [10, 1])
    imgs = imgs.reshape([1, 10, 10])

    sli = central_slice(imgs, size=[8, 8])
    assert np.all(sli == imgs[:, 1:-1, 1:-1])

    sli = central_slice(imgs, size=[7, 7])
    assert np.all(sli == imgs[:, 2:-1, 2:-1])

    sli = central_slice(imgs, size=[9, 9])
    assert np.all(sli == imgs[:, 1:, 1:])


def test_central_pad():

    ones = np.ones([2, 3, 3])

    pad = central_pad(ones, [5, 5])
    res = np.zeros([2, 5, 5])
    res[:, 1:-1, 1:-1] += ones
    assert np.all(pad == res)

    pad = central_pad(ones, [4, 4])
    res = np.zeros([2, 4, 4])
    res[:, 1:, 1:] += ones
    assert np.all(pad == res)

    pad = central_pad(ones, [6, 6])
    res = np.zeros([2, 6, 6])
    res[:, 2:-1, 2:-1] += ones
    assert np.all(pad == res)
