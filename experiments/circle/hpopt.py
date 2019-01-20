from datetime import datetime
import pathlib
import logging

import numpy as np
import skopt
from skopt.utils import use_named_args
from skopt.space import Real, Integer

import torsk
from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
from torsk.data.utils import gauss2d_sequence, mackey_sequence, normalize
from torsk.hpopt import esn_tikhonov_fitnessfunc


logging.basicConfig(level="DEBUG")


opt_steps = 50
tik_steps = 20
tik_start = -5
tik_stop = 2
tik_betas = np.logspace(tik_start, tik_stop, tik_steps)
level2 = [{"tikhonov_beta":beta, "train_method":"tikhonov"} for beta in tik_betas]
level2.append({"tikhonov_beta":None, "train_method":"pinv_lstsq"})

output_dir = pathlib.Path("hpopt")

dimensions = [
    Real(low=0.5, high=2.0, name="spectral_radius"),
]

starting_params = [
    1.0,    # esn_spectral_radius
]

params = torsk.Params("params.json")
params.density = 0.01
params.input_map_specs = [
    {"type": "random_weights", "size": [10000], "input_scale": 0.25}
]
t = np.arange(0, 200*np.pi, 0.1)
#x, y = np.sin(t), np.cos(0.3 * t)
x, y = np.sin(0.3 * t), np.cos(t)
x = normalize(mackey_sequence(N=t.shape[0])) * 2 - 1

center = np.array([y, x]).T
images = gauss2d_sequence(center, sigma=0.5, size=params.input_shape)
dataset = ImageDataset(images, params, scale_images=True)

fitness = esn_tikhonov_fitnessfunc(
    output_dir, dataset, params, dimensions, level2, nr_avg_runs=3)


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
