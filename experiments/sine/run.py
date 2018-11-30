import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import torsk
from torsk.models import ESN
from torsk.visualize import plot_mackey
from torsk.data import SineDataset, SeqDataLoader


# Parameters and model initialization
params = torsk.Params("params.json")

dataset = SineDataset(
    train_length=params.train_length,
    pred_length=params.pred_length)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

model = ESN(params)

predictions, labels = [], []
for i in tqdm(range(30)):
    model, outputs, pred_labels, _ = torsk.train_predict_esn(
        model=model, loader=loader, params=params)
    predictions.append(outputs.squeeze().numpy())
    labels.append(pred_labels.squeeze().numpy())

predictions, labels = np.array(predictions), np.array(labels)

plot_mackey(predictions, labels, weights=model.out.weight.numpy())
plt.show()
