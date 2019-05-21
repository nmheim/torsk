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
    {"type": "pixels", "size": [40, 40], "input_scale": 2., "flatten":True},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [25, 25], "kernel_type":"gauss",
            "input_scale": 1., "flatten":False},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [15, 15], "kernel_type":"gauss",
            "input_scale": 1., "flatten":False},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [ 5, 5], "kernel_type":"random",
            "input_scale": 1., "flatten":False},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [10, 10], "kernel_type":"random",
            "input_scale": 1., "flatten":False},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations":[
        {"type": "conv", "mode": "same", "size": [20, 20],
            "kernel_type":"random", "input_scale": 1., "flatten":False},
        {"type": "pixels", "size":[30,30], "flatten": True},
    ]},
    {"type": "compose", "operations": [
        {"type": "gradient", "input_scale": 1., "flatten": False},
        {"type": "pixels", "size": [50,25], "flatten": True},
    ]},
    {"type": "random_weights", "size": [3000], "input_scale": 0.025},
    {"type": "dct", "size": [10,10], "input_scale": 0.1, "flatten":True},
]

params.spectral_radius = 2.5
params.density = 0.01
params.input_shape = [50,50]
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

if params.backend == "numpy":
    logger.info("Running with NUMPY backend")
    from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
else:
    logger.info("Running with TORCH backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN


npypath = pathlib.Path("Kuro_SSH_5daymean.npy")
images = np.load(npypath)[:, 90:190, 90:190]
images[images>10000.] = 0.
images = resample2d_sequence(images, params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, "/tmp/torsk_experiments/kuro_conv_5daymean100x100",
    steps=1, step_length=1)

from torsk.imed import imed_metric
print(imed_metric(outputs, pred_labels)[25])
anim = animate_double_imshow(outputs, pred_labels)
plt.show()
