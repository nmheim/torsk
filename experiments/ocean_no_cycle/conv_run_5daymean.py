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
    {"type": "pixels", "size": [30, 30], "input_scale": 3.},
    {"type": "conv", "mode": "same", "size": [5, 5], "kernel_type":"gauss", "input_scale": 2.},
    {"type": "conv", "mode": "same", "size": [10,10], "kernel_type":"gauss", "input_scale": 1.5},
    {"type": "conv", "mode": "same", "size": [15, 15], "kernel_type":"gauss", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [ 5, 5], "kernel_type":"random", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    {"type": "conv", "mode": "same", "size": [20, 20], "kernel_type":"random", "input_scale": 1.},
    {"type": "dct", "size": [15, 15], "input_scale": 1.},
    {"type": "gradient", "input_scale": 1.},
    {"type": "gradient", "input_scale": 1.}
]

params.spectral_radius = 1.5
params.density = 0.05
params.input_shape = [30, 30]
params.train_length = 12*73
params.pred_length = 73
params.transient_length = 3*73
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_lstsq"
params.tikhonov_beta = 3e1
params.debug = False
params.imed_loss = True
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


npzpath = pathlib.Path("/home/niklas/erda_save/Ocean/esn/Kuro_SSH_5daymean_detrended.npz")
images = np.load(npzpath)["arr_3"][:, 90:190, 90:190]
images = resample2d_sequence(images, params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, "/home/niklas/erda_save/kuro_conv_5daymean_detrended_d005",
    steps=1000, step_length=1)
