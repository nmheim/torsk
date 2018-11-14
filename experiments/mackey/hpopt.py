from datetime import datetime
import pathlib
import torch
import skopt
from skopt.utils import use_named_args
from skopt.space import Real

from torsk.models import ESN
from torsk.utils import Params
from torsk.data import MackeyDataset, SeqDataLoader


opt_steps = 100
output_dir = pathlib.Path("hpopt")

dimensions = [
    Real(low=0.5, high=2.0, name="spectral_radius"),
    Real(low=0.01, high=2.0, name="in_weight_init"),
    Real(low=0.01, high=2.0, name="in_bias_init"),
    #Real(low=1e-5, high=1e1, name="tikhonov_beta", prior="log_scale")
]

starting_params = [
    1.3,    # esn_spectral_radius
    0.5,    # in_weight_init
    0.5,    # in_bias_init
    #1.0,    # tikhonov_beta
]

params = Params("hpopt_params.json")
train_length = params.train_length
pred_length = params.pred_length
transient_length = params.transient_length

# input/label setup
dataset = MackeyDataset(
    seq_length=train_length + pred_length,
    simulation_steps=train_length + pred_length + opt_steps * 10)
loader = iter(SeqDataLoader(dataset, batch_size=1, shuffle=True))


@use_named_args(dimensions=dimensions)
def fitness(**sampled_params):

    params.update(sampled_params)
    print(f"Current model parameters {params}")
    model = ESN(params)


    error = []
    for _ in range(5):
        inputs, labels = next(loader)
        train_inputs, train_labels = inputs[:train_length], labels[:train_length]
        test_inputs = inputs[train_length - transient_length:train_length]
        test_labels = labels[train_length - transient_length:]
        
        # create states and train
        state = torch.zeros(1, params.hidden_size)
        _, states = model(train_inputs, state)
        model.train(
            inputs=train_inputs[transient_length:, 0],
            states=states[transient_length:, 0],
            labels=train_labels[transient_length:, 0],
            method=params.train_method,
            beta=params.tikhonov_beta)
         
        # predict
        state = torch.zeros(1, params.hidden_size)
        outputs, _ = model(test_inputs, state, nr_predictions=pred_length)

        err = (test_labels[transient_length:] - outputs[transient_length:])**2
        error.append(err)
    metric = torch.mean(torch.cat(error, dim=0))
    if not torch.isfinite(metric):
        metric = 1e6
    else:
        metric = metric.item()

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

