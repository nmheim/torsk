import sys
import logging
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import torsk
from torsk.visualize import animate_double_imshow


params = torsk.Params()
params.input_map_specs = [
    {"type": "pixels", "size": [30, 30], "input_scale": 6.},
    # {"type": "pixels", "size": [25, 25], "input_scale": 6.},
    # {"type": "pixels", "size": [20, 20], "input_scale": 6.},
    # {"type": "pixels", "size": [15, 15], "input_scale": 6.},
    # {"type": "pixels", "size": [10, 10], "input_scale": 6.},
    # {"type": "pixels", "size": [5, 5], "input_scale": 6.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 3.},
    {"type": "conv", "size": [30, 30], "kernel_type":"gauss", "input_scale": 0.05},
    {"type": "conv", "size": [30, 30], "kernel_type":"random", "input_scale": 0.25},
    # {"type": "conv", "size": [30, 30], "kernel_type":"random", "input_scale": 0.25},
    # {"type": "dct", "size": [50, 50], "input_scale": 1.0},
    # {"type": "dct", "size": [10, 10], "input_scale": 1.},
    # {"type": "random_weights", "size": [2000], "weight_scale": 0.125}
]
params.spectral_radius = 1.5
params.density = 0.001
params.input_shape = [100, 100]
params.train_length = 1000
params.pred_length = 300
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv"
params.tikhonov_beta = 3e1
params.debug = True
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

logger.info(params)

logger.info("Loading ...")
ncpath = '../../data/ocean/kuro_SSH_3daymean_scaled.nc'
with nc.Dataset(ncpath, 'r') as src:
    images = src["SSH"][:, 90:190, 90:190]
    images, mask = images.data, images.mask
    images[mask] = 0.
dataset = ImageDataset(images, params)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(model, dataset)

logger.info("Visualizing results ...")

if params.backend == "torch":
    real_pixels = pred_labels.squeeze().numpy()
    predicted_pixels = outputs.squeeze().numpy()
else:
    real_pixels = pred_labels
    predicted_pixels = outputs

y, x = 5, 5
plt.plot(real_pixels[:, y, x])
plt.plot(predicted_pixels[:, y, x])
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

