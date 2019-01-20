import pathlib
import logging

import numpy as np
import netCDF4 as nc
import pandas as pd
from skopt.utils import use_named_args

import torsk
from torsk.models.numpy_esn import NumpyESN

logger = logging.getLogger(__name__)


SECOND_LEVEL_HYPERPARAMETERS = [
    "transient_length", "pred_length", "tikhonov_beta", "train_method"
]


def is_second_level_param(param_name):
    return param_name in SECOND_LEVEL_HYPERPARAMETERS


def valid_second_level_params(params):
    for name in params:
        if not is_second_level_param(name):
            return False
    return True


def create_path(root, param_dict, prefix=None, postfix=None):
    if not isinstance(root, pathlib.Path):
        root = pathlib.Path(root)
    folder = torsk._fix_prefix(prefix)
    for key, val in param_dict.items():
        folder += f"{key}:{val}-"
    folder = folder[:-1]
    if postfix is not None:
        folder += f"-{postfix}"
    return root / folder


def parse_path(path):
    param_dict = {}
    for string in path.name.split("-"):
        if ":" in string:
            key, val = string.split(":")
            try:
                val = eval(val)
            except Exception:
                pass
            param_dict[key] = val
    return param_dict


def get_hpopt_dirs(rootdir):
    """Yields all subdirectories that contain a trained-model.pth"""
    if not isinstance(rootdir, pathlib.Path):
        rootdir = pathlib.Path(rootdir)

    for trained_path in rootdir.glob("**"):
        if trained_path.joinpath("trained-model.pth").exists():
            yield trained_path


def get_metric(trained_model_dir):
    with nc.Dataset(trained_model_dir / "pred_data.nc", "r") as src:
        metric = src["metric"][:]
    return metric.mean()


def optimize_wout(outdir, model, inputs, states, train_labels, pred_labels):

    tlen = model.params.transient_length
    plen = model.params.pred_length
    init_input = train_labels[-1]
    init_state = states[-1]

    logger.debug(f"Optimizing Wout with method:{model.params.train_method}"
                 f" / beta:{model.params.tikhonov_beta}")
    model.optimize(
        inputs=inputs[tlen:], states=states[tlen:], labels=train_labels[tlen:])

    logger.debug(f"Predicting {plen} steps")
    outputs, out_states = model.predict(
        init_input, init_state, nr_predictions=plen)

    pred_data_nc = outdir / "pred_data.nc"
    logger.debug(f"Dumping prediction data at {pred_data_nc}")
    torsk.dump_prediction(
        pred_data_nc, outputs=outputs, labels=pred_labels, states=out_states)

    logger.debug(f"Dumping model at {outdir}")
    torsk.save_model(outdir, model, prefix="trained")


def _evaluate_hyperparams(outdir, model, dataset, level2_params_list):
    logger.debug(f"Dumping model at {outdir}")
    torsk.save_model(outdir, model, prefix="untrained")

    idx = np.random.randint(low=0, high=len(dataset))
    logger.debug(f"Getting inputs/labels from dataset index: {idx}")
    inputs, labels, pred_labels = dataset[idx]

    logger.debug(f"Creating {inputs.shape[0]} training states")
    zero_state = np.zeros([model.esn_cell.hidden_size], dtype=model.esn_cell.dtype)
    _, states = model.forward(inputs, zero_state, states_only=True)

    train_data_nc = outdir / "train_data.nc"
    logger.debug(f"Dumping training data at {train_data_nc}")
    torsk.dump_training(
        train_data_nc,
        inputs=inputs, labels=labels, states=states,
        pred_labels=pred_labels)

    for level2 in level2_params_list:
        if valid_second_level_params(level2):
            model.params.update(level2)
            level2_outdir = create_path(outdir, level2, prefix="trained")
            optimize_wout(
                outdir=level2_outdir, model=model, inputs=inputs,
                states=states, train_labels=labels, pred_labels=pred_labels)
        else:
            raise ValueError(
                f"Not all level2-params are of second level: {level2}")


def evaluate_hyperparams(outdir, params, level2_params_list, dataset, iters=1):

    if not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    for ii in range(1, iters + 1):

        logger.info("Building model")
        model = NumpyESN(params)

        rundir = outdir.joinpath(f"run-{ii:03d}")
        logger.info(f"Evaluation run: {rundir}")

        _evaluate_hyperparams(
            outdir=rundir, model=model, dataset=dataset,
            level2_params_list=level2_params_list)


def esn_tikhonov_fitnessfunc(outdir, dataset, params, dimensions,
                             level2_params_list, nr_avg_runs=10):
    """Fitness function for hyper-parameter optimization that automatically
    uses the best regularization parameter out of a given list of tikhonov_betas

    Parameters
    ----------
    outdir : pathlib.Path
        output directory
    dataset : torch.utils.DataLoader
        loads the necessary inputs/labels/pred_labels
    params : torsk.Params
        model/training parameters
    dimensions : list of skopt.Dimesion
        defining the hyper-parameter search space
    level2_params_list : list(dict)
        second level hyper-parameters to test for each sampled params set

    Returns
    -------
    fitness : function
        function to be passed to skopt.gp_minimize
    """

    @use_named_args(dimensions=dimensions)
    def fitness(**sampled_params):

        params.update(sampled_params)
        logger.info(f"Sampled params: {params}")

        subdir = create_path(outdir, sampled_params)

        evaluate_hyperparams(
            outdir=subdir, params=params, level2_params_list=level2_params_list,
            dataset=dataset, iters=nr_avg_runs)

        trained_model_dirs = get_hpopt_dirs(subdir)
        runs = []
        for model_dir in trained_model_dirs:
            data = {"level2": model_dir.name}
            data["metric"] = get_metric(model_dir)
            data["run"] = int(model_dir.parent.name.split("-")[-1])
            runs.append(data)

        dataframe = pd.DataFrame(runs)
        grouped = dataframe.groupby("level2").median()
        return grouped["metric"].min()
    return fitness
