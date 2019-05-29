import sys
import logging
import pathlib
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import torsk
from torsk.imed import imed_metric
from torsk.data.utils import resample2d_sequence
from torsk.visualize import animate_double_imshow

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "pixels", "size": [60, 60], "input_scale": 3., "flatten":True},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [5, 5], "kernel_type":"gauss", "input_scale": 2.},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [10, 10], "kernel_type":"gauss", "input_scale": 1.5},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [15, 15], "kernel_type":"gauss", "input_scale": 1.},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [ 5, 5], "kernel_type":"random", "input_scale": 1.},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [20, 20], "kernel_type":"random", "input_scale": 1.},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations": [
        {"type": "gradient", "input_scale": 1.},
        {"type": "pixels", "size": [80,40], "flatten": True},
    ]},
    {"type": "dct", "size": [60,60], "input_scale": 1.},
]

params.spectral_radius = 1.5
params.density = 0.01
params.input_shape = [100, 100]
params.train_length = 12*73
params.pred_length = 73
params.transient_length = 3*73
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_lstsq"
params.tikhonov_beta = 3e1
params.debug = False
params.imed_loss = False
params.update(sys.argv[1:])

logger.info(params)

if params.backend == "numpy":
    logger.info("Running with NUMPY backend")
    from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
else:
    logger.info("Running with TORCH backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN


npypath = pathlib.Path("/home/niklas/erda_save/Ocean/esn/Kuro_SSH_5daymean.npy")
images = np.load(npypath)[:, 90:190, 90:190]
images[images>10000.] = 0.
images = resample2d_sequence(images, params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, "/home/niklas/erda_save/kuro_conv_5daymean100x100",
    steps=1, step_length=1)

print(imed_metric(outputs, pred_labels)[25])
