import sys
import logging
import pathlib
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import torsk
from torsk.imed import imed_metric
from torsk.visualize import animate_double_imshow

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "pixels", "size": [30, 30], "input_scale": 3.},
    # {"type": "pixels", "size": [25, 25], "input_scale": 6.},
    #{"type": "pixels", "size": [20, 20], "input_scale": 2.},
    # {"type": "pixels", "size": [15, 15], "input_scale": 6.},
    # {"type": "pixels", "size": [10, 10], "input_scale": 6.},
    # {"type": "pixels", "size": [5, 5], "input_scale": 6.},
    # {"type": "conv", "size": [1, 1], "kernel_type":"gauss", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [2, 2], "kernel_type":"gauss", "input_scale": 2.},
    {"type": "conv", "mode": "same", "size": [5, 5], "kernel_type":"gauss", "input_scale": 2.},
    {"type": "conv", "mode": "same", "size": [10,10], "kernel_type":"gauss", "input_scale": 1.5},
    {"type": "conv", "mode": "same", "size": [15, 15], "kernel_type":"gauss", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [20, 20], "kernel_type":"gauss", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [25, 25], "kernel_type":"gauss", "input_scale": 1.},        
    {"type": "conv", "mode": "same", "size": [ 5, 5], "kernel_type":"random", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},        
    {"type": "conv", "mode": "same", "size": [20, 20], "kernel_type":"random", "input_scale": 1.},    
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"random", "input_scale": 1.},
     {"type": "dct", "size": [15, 15], "input_scale": 1.},
     {"type": "dct", "size": [15, 15], "input_scale": 1.},    
    # {"type": "dct", "size": [10, 10], "input_scale": 1.},
#    {"type": "random_weights", "size": [2000], "weight_scale": 1, "input_scale":0.05}
#    {"type": "random_weights", "size": [30*30], "weight_scale": 1, "input_scale":0.025}    
]

params.spectral_radius = 2.
params.density = 0.01
params.input_shape = [30, 30]
params.train_length = 800
params.pred_length = 300
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_svd"
params.tikhonov_beta = 0.01
params.imed_loss = True
params.debug = False

logger.info(params)

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
    model, dataset, "/mnt/data/toto/ocean/raw_conv_output",
    steps=100, step_length=10)
