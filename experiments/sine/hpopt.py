from datetime import datetime
import pathlib
import logging

import numpy as np
import skopt
from skopt.utils import use_named_args
from skopt.space import Real, Integer

import torsk
from torsk.data import SineDataset, SeqDataLoader
from torsk.hpopt import esn_tikhonov_fitnessfunc


logging.basicConfig(level="WARNING")


opt_steps = 50
tik_steps = 20
tik_start = -5
tik_stop = 2
tik_betas = np.logspace(tik_start, tik_stop, tik_steps)
level2 = [{"tikhonov_beta":beta, "train_method":"tikhonov"} for beta in tik_betas]
level2.append({"tikhonov_beta":None, "train_method":"pinv"})

output_dir = pathlib.Path("hpopt_output")

dimensions = [
    Real(low=0.5, high=2.0, name="spectral_radius"),
    Real(low=0.0, high=2.0, name="in_weight_init"),
]

starting_params = [
    1.0,    # esn_spectral_radius
    0.01,   # in_weight_init
]

params = torsk.Params("hpopt_params.json")
dataset = SineDataset(
    train_length=params.train_length,
    pred_length=params.pred_length)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

fitness = esn_tikhonov_fitnessfunc(
    output_dir, loader, params, dimensions, level2, nr_avg_runs=3)

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
