import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

import torsk
from torsk.data.utils import gauss2d_sequence, mackey_sequence, normalize
from torsk.visualize import animate_double_imshow


np.random.seed(0)

params = torsk.Params()
params.input_map_specs = [
    {"type": "random_weights", "size": [10000], "input_scale": 0.25}
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
params.train_method = "pinv_svd"
params.tikhonov_beta = 0.01
params.imed_loss = True
params.debug = False
params.cycle_length = 100

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
t = np.arange(0, 200*np.pi, 0.02*np.pi)
# x, y = np.sin(0.3 * t), np.cos(t)
x, y = np.sin(t), np.cos(0.3 * t)
# y = normalize(mackey_sequence(N=t.shape[0])) * 2 - 1

center = np.array([y, x]).T
images = gauss2d_sequence(center, sigma=0.5, size=params.input_shape)
dataset = ImageDataset(images, params)

logger.info("Building model ...")
model = ESN(params)

logger.info("Training + predicting ...")
model, outputs, pred_labels = torsk.train_predict_esn(
    model, dataset, outdir="sin_cos03_output", steps=1, step_length=11)

logger.info("Visualizing results ...")
from torsk.imed import imed_metric

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

plt.plot(imed_metric(real_pixels, predicted_pixels))
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

