from datetime import datetime
import pathlib
import logging

import numpy as np
import skopt
from skopt.utils import use_named_args
from skopt.space import Real, Integer

import torsk
from torsk.data import NetcdfDataset, SeqDataLoader
from torsk.hpopt import esn_tikhonov_fitnessfunc


logging.basicConfig(level="INFO")


opt_steps = 50
tik_steps = 20
tik_start = -5
tik_stop = 2
tikhonov_betas = np.logspace(tik_start, tik_stop, tik_steps)

output_dir = pathlib.Path("hpopt")

dimensions = [
    Real(low=0.5, high=2.0, name="spectral_radius"),
    Real(low=0.0, high=2.0, name="in_weight_init"),
    Real(low=0.0, high=2.0, name="in_bias_init"),
]

starting_params = [
    1.0,    # esn_spectral_radius
    1.0,    # in_weight_init
    1.0,    # in_bias_init
]

Nx, Ny = 30, 30
params = torsk.Params("params.json")
params.input_size  = Nx*Ny
params.output_size = Nx*Ny
params.train_length = 1500
params.hidden_size = 10000
params.pred_length = 300
params.train_length = 1500
params.train_method = "tikhonov"

ncpath = pathlib.Path('../../data/ocean/kuro_SSH_3daymean_scaled.nc')
dataset = NetcdfDataset(ncpath, params.train_length, params.pred_length,
    xslice=slice(90, 190), yslice=slice(90, 190), size=[Nx, Ny])
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))

fitness = esn_tikhonov_fitnessfunc(loader, params, dimensions, tikhonov_betas)

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
