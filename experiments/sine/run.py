import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import torsk
from torsk.models import ESN
from torsk.visualize import plot_mackey
from torsk.data import SineDataset, SeqDataLoader
from torsk import utils


logging.basicConfig(level="INFO")


# Parameters and model initialization
params = torsk.Params("params.json")
print(params)

dataset = SineDataset(
    train_length=params.train_length,
    pred_length=params.pred_length)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))


model = ESN(params)
inputs, train_labels, pred_labels, orig_data = next(loader)

zero_state = torch.zeros(1, params.hidden_size)
_, states = model(inputs, zero_state, states_only=True)

predictions, labels = [], []
for i in tqdm(range(30)):
    model = ESN(params)
    model, outputs, pred_labels, _ = torsk.train_predict_esn(
        model=model, loader=loader, params=params)
    predictions.append(outputs.squeeze().numpy())
    labels.append(pred_labels.squeeze().numpy())

predictions, labels = np.array(predictions), np.array(labels)
error = (predictions - labels)**2
print(f"Metric: {error.mean()}")

plot_mackey(predictions, labels, weights=model.out.weight.numpy())
plt.show()
