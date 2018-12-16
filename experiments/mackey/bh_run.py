import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

import torsk
from torsk.models.bh_esn import ESN, train_predict_esn
from torsk.data import MackeyDataset, SeqDataLoader
from torsk.visualize import plot_mackey


sns.set_style("whitegrid")

params = torsk.Params("params.json")
params.train_method = "pinv"

train_length = params.train_length
pred_length = params.pred_length
transient_length = params.transient_length
beta = params.tikhonov_beta
print(params)

dataset = MackeyDataset(train_length, pred_length, simulation_steps=3000)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

model = ESN(params)

predictions, labels = [], []
for i in tqdm(range(20)):

    model, outputs, pred_labels, _ = train_predict_esn(
        model=model, loader=loader, params=params, outdir=".")

    predictions.append(outputs.squeeze())
    labels.append(pred_labels.squeeze())

predictions, labels = np.array(predictions), np.array(labels)

plot_mackey(predictions, labels, weights=model.wout)
plt.show()
