import sys
import logging
import pathlib

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns

import torsk
from torsk.data.utils import mackey_anomaly_sequence, normalize
from torsk.visualize import animate_double_imshow

sns.set_style("whitegrid")
np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "random_weights", "size": [1000], "input_scale": 1.}
]
params.spectral_radius = 1.5
params.density = 0.05
params.input_shape = [1, 1]
params.train_length = 2200
params.pred_length = 500
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "dense"
params.backend = "numpy"
params.train_method = "pinv_svd"
params.tikhonov_beta = 2.0
params.debug = False
params.imed_loss = False
params.anomaly_start = 2400
params.anomaly_step = 300
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

logger.info("Building model ...")
model = ESN(params)

mackey, _ = mackey_anomaly_sequence(
    N=3700,
    anomaly_start=params.anomaly_start,
    anomaly_step=params.anomaly_step)
mackey = normalize(mackey) * 2 - 1
mackey = mackey[:, np.newaxis, np.newaxis]
dataset = ImageDataset(mackey, params, scale_images=False)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, "mackey_anomaly_output",
    steps=1, step_length=2)

plt.plot(pred_labels[:,0,0], label="Prediction")
plt.plot(outputs[:,0,0], label="Truth")
plt.legend()
plt.show()
