import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

import torch
import torsk
from torsk.models import ESN
from torsk.data import CircleDataset, SeqDataLoader
from torsk.visualize import animate_double_imshow
import sys

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)

def update_params(params,args):
    for i in range(0,len(args),2):
        key,value = args[i],args[i+1];
        print(key,"eq",value)
        params.dict[key] = eval(value)

params = torsk.Params("params.json")
update_params(params,sys.argv[1:]);

if(params.domain == "DCT"):
    params.size = params.ksize;
else:
    params.size = params.xsize;


params.input_size = params.size[0]*params.size[1];
params.output_size = params.size[0]*params.size[1];

logger.info(params)

logger.info("Creating circle dataset ...")
x = np.sin(np.arange(0, 200*np.pi, 0.1))
y = np.cos(0.25 * np.arange(0, 200*np.pi, 0.1))
center = np.array([y, x]).T
sigma = params.sigma

#TODO: Just pass params-parameter instead of extracting everything as arguments?
dataset = CircleDataset(
    params.train_length, params.pred_length,
    center=center, sigma=sigma, xsize=params.xsize,ksize=params.ksize,domain=params.domain)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))


logger.info("Building model ...")
model = ESN(params)


logger.info("Training + predicting ...")
model, outputs, pred_labels, _ = torsk.train_predict_esn(
    model, loader, params, outfile="results.nc", modelfile=None)

logger.info("Visualizing results ...")

weight = model.esn_cell.res_weight._values().numpy()
hist, bins = np.histogram(weight, bins=100)
plt.plot(bins[1:], hist)
plt.show()

real_pixels      = dataset.to_image(pred_labels);
predicted_pixels = dataset.to_image(outputs);

y, x = 10, 10
plt.plot(real_pixels[:, y, x])
plt.plot(predicted_pixels[:, y, x])
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

