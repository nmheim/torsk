import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

import torsk
from torsk.models.numpy_esn import ESN
from torsk.utils import gauss2d
from torsk.data.image import NumpyImageDataset
from torsk.visualize import animate_double_imshow

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def update_params(params,args):
    for i in range(0,len(args),2):
        key,value = args[i],args[i+1];
        print(key,"eq",value)
        params.dict[key] = eval(value)


#params = torsk.Params("params.json")
params = torsk.default_params()
params.feature_specs = [{"type": "pixels", "size": [10, 10]}]
params.in_weight_init = 1.0
params.in_bias_init = 1.0
params.spectra_radius = 2.0
params.hidden_size = 2000
params.input_size = 100
params.train_length = 800
#update_params(params,sys.argv[1:]);

logger.info(params)

logger.info("Creating circle dataset ...")
x = np.sin(np.arange(0, 200*np.pi, 0.1))
y = np.cos(0.25 * np.arange(0, 200*np.pi, 0.1))
center = np.array([y, x]).T
images = gauss2d(center, sigma=0.5, size=size)
dataset = NumpyImageDataset(images, params)

anim = animate_double_imshow(images, images)
plt.show()

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

