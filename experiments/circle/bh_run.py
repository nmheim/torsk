import logging
import pathlib
import bohrium as np
#import numpy as np
import matplotlib.pyplot as plt
from bh_junk import *
import sys

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)

def update_params(params,args):
    for i in range(0,len(args),2):
        key,value = args[i],args[i+1];
        print(key,"eq",value)
        params.dict[key] = eval(value)

params = Params("params.json")
update_params(params,sys.argv[1:]);

if(params.domain == "DCT"):
    params.size = params.ksize;
else:
    params.size = params.xsize;


params.input_size = params.size[0]*params.size[1];
params.output_size = params.size[0]*params.size[1];

logger.info(params)


logger.info("Creating circle dataset ...")
t = np.linspace(0, 200*np.pi, 2000)
x = np.sin(t)
y = np.cos(0.25 * t)
center = np.array([y, x]).T
sigma = params.sigma

#TODO: Just pass params-parameter instead of extracting everything as arguments?
dataset = CircleDataset(params, center=center, sigma=sigma)


logger.info("Building model ...")
model = ESN(params)


logger.info("Training + predicting ...")
model, outputs, pred_labels, _ = train_predict_esn(model, dataset, params, outdir=".")

logger.info("Visualizing results ...")

# weight = model.esn_cell.weight_hh.values
# hist, bins = np.histogram(weight, bins=100)
# plt.plot(bins[1:], hist)
# plt.show()

real_pixels      = dataset.to_image(pred_labels);
predicted_pixels = dataset.to_image(outputs);

y, x = 10, 10
plt.plot(real_pixels[:, y, x])
plt.plot(predicted_pixels[:, y, x])
plt.show()

anim = animate_double_imshow(real_pixels,predicted_pixels)
plt.show()

