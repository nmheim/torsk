from datetime import datetime
import pathlib
import numpy as np
import torch
import skopt
from skopt.utils import use_named_args
from skopt.space import Real

import torsk
from torsk.models import ESN
from torsk.utils import Params
from torsk.data import SineDataset, SeqDataLoader


opt_steps = 50
output_dir = pathlib.Path("hpopt")

dimensions = [
    Real(low=0.5, high=2.0, name="spectral_radius"),
    Real(low=0.001, high=2.0, name="in_weight_init"),
    Real(low=0.001, high=2.0, name="in_bias_init"),
    Real(low=1e-5, high=1e1, name="tikhonov_beta", prior="log_scale")
]

starting_params = [
    1.0,    # esn_spectral_radius
    0.01,    # in_weight_init
    0.01,    # in_bias_init
    0.01,    # tikhonov_beta
]

params = Params("hpopt_params.json")

dataset = SineDataset(
    train_length=params.train_length,
    pred_length=params.pred_length)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))


@use_named_args(dimensions=dimensions)
def fitness(**sampled_params):

    params.update(sampled_params)
    print(f"Current model parameters {params}")
    model = ESN(params)


    predictions, labels = [], []
    for _ in range(30):
        model, outputs, pred_labels = torsk.train_predict_esn(
            model=model, loader=loader, params=params)
        predictions.append(outputs.squeeze().numpy())
        labels.append(pred_labels.squeeze().numpy())

    predictions, labels = np.array(predictions), np.array(labels)
    error = (predictions - labels)**2

    metric = np.mean(error)
    if not np.isfinite(metric):
        metric = 1e6
    return metric


if __name__ == "__main__":
    
    # TODO: add callback that saves checkpoints
    result = skopt.gp_minimize(
        n_calls=opt_steps,
        func=fitness,
        dimensions=dimensions,
        acq_func="gp_hedge",
        x0=starting_params,
        verbose=True)

    print("\n\nBest parameters:")
    keys = [d.name for d in dimensions]
    for key, val in zip(keys, result.x):
        print("\t"+key, val)
    print("With loss:", result.fun)
    print("\n")

    sorted_results = sorted(zip(result.func_vals, result.x_iters))
    for res in sorted_results:
        print(res)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    skopt.dump(
        result,
        output_dir.joinpath(f"result_{now}.pkl"),
        store_objective=False)

