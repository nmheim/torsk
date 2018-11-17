import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

import torsk
from torsk.models import ESN
from torsk.utils import Params
from torsk.data import MackeyDataset, SeqDataLoader
from torsk.visualize import plot_mackey


sns.set_style("whitegrid")

params = Params("params.json")
train_length = params.train_length
pred_length = params.pred_length
transient_length = params.transient_length
beta = params.tikhonov_beta
method = params.train_method
print(params)

dataset = MackeyDataset(train_length, pred_length, simulation_steps=3000)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

model = ESN(params)

predictions, labels = [], []
for i in tqdm(range(20)):
    model, outputs, pred_labels = torsk.train_predict_esn(
        model=model, loader=loader, params=params)
    predictions.append(outputs.squeeze().numpy())
    labels.append(pred_labels.squeeze().numpy())


predictions, labels = np.array(predictions), np.array(labels)

plot_mackey(predictions, labels, weights=model.out.weight.numpy())
plt.show()
