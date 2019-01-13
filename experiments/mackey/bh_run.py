#import numpy as np
import bohrium as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from bh_junk import *

print("Assigning BH array")
a = np.linspace(0,1,100);
print("Printing BH array")
print(a)

import pathlib
import logging

import numpy as np
#import bohrium as np
import scipy as sp


_module_dir = pathlib.Path(__file__).absolute().parent
logger = logging.getLogger(__name__)

    
sns.set_style("whitegrid")

params = Params("params.json")
#params.train_method = "pinv"

train_length = params.train_length
pred_length = params.pred_length
transient_length = params.transient_length
beta = params.tikhonov_beta
print(params)

dataset = MackeyDataset(train_length, pred_length, simulation_steps=3000)

model = ESN(params)

predictions, labels = [], []
for i in tqdm(range(5)):

    model, outputs, pred_labels, _ = train_predict_esn(
        model=model, loader=dataset, params=params, outdir=".")

    predictions.append(outputs.squeeze())
    labels.append(pred_labels.squeeze())

print(predictions)
print(labels)
print("Completed predictions")
predictions, labels = np.array(predictions), np.array(labels)

print("Copied to BH")
print("Plotting")
plot_mackey(predictions, labels, weights=model.wout)
print("Show()")
plt.show()


