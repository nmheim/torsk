import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

import torsk
from torsk.data.utils import gauss2d_sequence
from torsk.visualize import animate_double_imshow


np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "random_weights", "size": [10000], "input_scale": 0.125}
]

params.spectral_radius = 2.0
params.density = 0.001
params.input_shape = [30, 30]
params.train_length = 2000
params.pred_length = 300
params.transient_length = 200
params.dtype = "float64"
params.reservoir_representation = "sparse"
params.backend = "numpy"
params.train_method = "pinv_lstsq"
params.imed_loss = False
params.tikhonov_beta = None
params.debug = False

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

logger.info("Creating circle dataset ...")
t = np.arange(0, 200 * np.pi, 0.02 * np.pi)
x, y = np.sin(0.3 * t), np.cos(t)

center = np.array([y, x]).T
images = gauss2d_sequence(center, sigma=0.5, size=params.input_shape)
dataset = ImageDataset(images, params)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, outdir="rand_output", steps=1)

logger.info("Visualizing results ...")
if params.backend == "torch":
    pred_labels = pred_labels.squeeze().numpy()
    outputs = outputs.squeeze().numpy()

anim = animate_double_imshow(pred_labels, outputs)
plt.show()
