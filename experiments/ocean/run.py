import sys
import logging
import pathlib

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import torsk
from torsk.visualize import animate_double_imshow

np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "random_weights", "size": [10000], "input_scale": 0.25}
]
params.spectral_radius = 2.0
params.density = 0.001
params.input_shape = [30, 30]
params.train_length = 1300
params.pred_length = 200
params.transient_length = 300
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_svd"
params.tikhonov_beta = 3e1
params.debug = False
params.imed_loss = True
params.update(sys.argv[1:])

logger = logging.getLogger(__file__)
level = "DEBUG" if params.debug else "INFO"
logging.basicConfig(level=level)
logging.getLogger("matplotlib").setLevel("INFO")

if params.backend == "numpy":
    logger.info("Running with NUMPY backend")
    from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
else:
    logger.info("Running with TORCH backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN


ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')

with nc.Dataset(ncpath, "r") as src:
    from torsk.data.utils import resample2d_sequence
    images = src["SSH"][:, 90:190, 90:190]
    images, mask = images.data, images.mask
    images[mask] = 0.
    images = resample2d_sequence(images, params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, "/home/niklas/erda_save/kuro_basicesn_withimed",
    steps=1000, step_length=1)
