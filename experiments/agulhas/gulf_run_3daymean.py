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


#input_root = "/mnt/data/avery/torsk_experiments/"
#output_root= "/mnt/data/avery/"
input_root = "./"
output_root= "./"

nx, ny = 43, 23
#nx, ny = 86,46
#nx, ny = 112, 60
input_file  =  "gulf_SSH_3daymean_downscaled.npz"
output_file = f"gulf_3daymean_LR{nx}x{ny}"

params = torsk.Params()
params.input_map_specs = [
    {"type": "pixels", "size": [ny, nx], "input_scale": 3.},
    {"type": "conv", "mode": "valid", "size": [5, 5], "kernel_type":"gauss", "input_scale": 2.},
    {"type": "conv", "mode": "valid", "size": [10,10], "kernel_type":"gauss", "input_scale": 1.5},
    {"type": "conv", "mode": "valid", "size": [15, 15], "kernel_type":"gauss", "input_scale": 1.},
    {"type": "conv", "mode": "valid", "size": [10, 10], "kernel_type":"random", "input_scale": 1.},
    {"type": "conv", "mode": "valid", "size": [20, 20], "kernel_type":"random", "input_scale": 1.},
    # {"type": "dct", "size": [15, 15], "input_scale": 1.},
    {"type": "dct", "size": [15, 15], "input_scale": 1.},
    {"type": "gradient", "input_scale": 1.},
    {"type": "gradient", "input_scale": 1.}
]

params.spectral_radius = 1.5
params.density = 0.01
params.input_shape = [ny,nx]
params.train_length = 10*122
params.pred_length = 122
params.transient_length = 2*122
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_svd"
params.tikhonov_beta = 3e1
params.debug = False
params.imed_loss = True
params.update(sys.argv[1:])
outdir = pathlib.Path(output_root+output_file)

logger.info(params)

if params.backend == "numpy":
    logger.info("Running with NUMPY backend")
    from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
    from torsk.models.numpy_esn import NumpyESN as ESN
else:
    logger.info("Running with TORCH backend")
    from torsk.data.torch_dataset import TorchImageDataset as ImageDataset
    from torsk.models.torch_esn import TorchESN as ESN


npzpath = pathlib.Path(input_root+input_file)
images = np.load(npzpath)["SSH"][:]
images = resample2d_sequence(images[:,:,:], params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

prefix = "idx0"
model_path = outdir.joinpath(f"{prefix}-model.pkl")
print(model_path)
if model_path.exists():
    logger.info("Restoring model ...")
    model = torsk.load_model(outdir, prefix=prefix)
else:
    logger.info("Building model ...")
    model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, outdir,
    steps=5*122, step_length=1,step_start=0)
