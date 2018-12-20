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


Nx, Ny = 30, 30
params = torsk.default_params()
params.feature_specs = [{"type": "pixels", "size": [Nx, Ny]}]
params.in_weight_init = 1.0
params.in_bias_init = 1.0
params.spectra_radius = 2.0
params.hidden_size = 2000
params.input_size = Nx * Ny
params.train_length = 800

params.train_method = "tikhonov"
params.tikhonov_beta = 3e1

update_params(params,sys.argv[1:]);

if params.backend == "numpy":
    logger.info("Running with Numpy backend")
    from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
elif params.backend == "torch":
    logger.info("Running with PyTorch backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN
    # TODO: fix dtypes !!!
    params.dtype = "float32"

logger.info(params)

logger.info("Loading + resampling of kuro window ...")
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

# weight = model.esn_cell.weight_hh._values().numpy()
# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

real_pixels      = dataset.to_images(pred_labels)
predicted_pixels = dataset.to_images(outputs)

y, x = 5, 5
plt.plot(real_pixels[:, y, x])
plt.plot(predicted_pixels[:, y, x])
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

