import netCDF4 as nc
import numpy as np
from torsk.data.ocean import NetcdfDataset


def test_ocean(tmpdir):

    ncpath = tmpdir.join('test.nc')
    with nc.Dataset(ncpath, 'w') as ncfile:
        ncfile.createDimension('time', 100)
        ncfile.createDimension('ydim', 10)
        ncfile.createDimension('xdim', 10)
        ssh = ncfile.createVariable('SSH', np.float32, ['time', 'ydim', 'xdim'])
        ssh[:] = np.random.uniform(size=[100, 10, 10])

    dataset = NetcdfDataset(
        ncpath, seq_length=30,
        xslice=slice(0, 10), yslice=slice(0, 10),
        size=[10, 10])
    inputs, labels = dataset[0]

    assert inputs.shape == (30, 100)
    assert labels.shape == (30, 100)
    assert np.all(inputs[1].numpy() == labels[0].numpy())
