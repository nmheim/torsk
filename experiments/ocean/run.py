import sys
import logging
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import torsk
from torsk.data.utils import gauss2d_sequence
from torsk.visualize import animate_double_imshow

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def update_params(params,args):
    for i in range(0,len(args),2):
        key,value = args[i],args[i+1];
        params.dict[key] = eval(value)


params = torsk.Params()
params.input_map_specs = [
    # {"type": "pixels", "size": [20, 20], "input_scale": 9.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"gauss", "input_scale": 9.},
    # {"type": "conv", "size": [5, 5], "kernel_type":"mean", "input_scale": 9.},
    # {"type": "dct", "size": [20, 20], "input_scale": 1.},
    {"type": "random_weights", "size": 2500, "weight_scale": 1.0}
]
params.spectral_radius = 1.5
params.density = 0.01
params.input_shape = [25, 25]
params.train_length = 1000
params.pred_length = 300
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "tikhonov"
params.tikhonov_beta = 3e1

update_params(params,sys.argv[1:]);

if params.backend == "numpy":
    logger.info("Running with Numpy backend")
    from torsk.data.numpy_dataset import NumpyRawImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
elif params.backend == "torch":
    raise NotImplementedError
    logger.info("Running with PyTorch backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN
    # TODO: fix dtypes !!!
    params.dtype = "float32"

logger.info(params)

logger.info("Loading ...")
ncpath = '../../data/ocean/kuro_SSH_3daymean_scaled.nc'
with nc.Dataset(ncpath, 'r') as src:
    images = src["SSH"][:, 90:190:4, 90:190:4]
    images, mask = images.data, images.mask
    images[mask] = 0.
dataset = ImageDataset(images, params)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(model, dataset)

logger.info("Visualizing results ...")

# weight = model.esn_cell.weight_hh._values().numpy()
# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

real_pixels = pred_labels
predicted_pixels = outputs

y, x = 5, 5
plt.plot(real_pixels[:, y, x])
plt.plot(predicted_pixels[:, y, x])
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

