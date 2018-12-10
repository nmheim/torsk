import pathlib
import logging

import numpy as np
from skopt.utils import use_named_args
import torch
from tqdm import tqdm

from torsk.models import ESN
from torsk import utils

logger = logging.getLogger(__name__)


SECOND_LEVEL_HYPERPARAMETERS = [
    "sigma", "domain", "transient_length", "pred_length"
]


def is_second_level_param(param_name):
    return param_name in SECOND_LEVEL_HYPERPARAMETERS


def valid_second_level_params(params):
    for name in params:
        if not is_second_level_param(name):
            return False
    return True


def evaluate_hyperparams(outdir, default_params, hyper_params, loader, tikhonov_betas):
    if not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logger.info("Building model")
    model = ESN(default_params)

    logger.info(f"Dumping model at {outdir}/[params.json, model.pth]")
    utils.save_model(outdir, model)

    logger.info("Loading dataset")
    inputs, labels, pred_labels, orig_data = next(loader)

    logger.info(f"Creating {inputs.size(0)} training states")
    states = utils.create_training_states(model, inputs)

    # TODO: save orig_data?
    train_data_nc = outdir / "train_data.nc"
    logger.info(f"Dumping training data at {train_data_nc}")
    utils.dump_training(
        fname=train_data_nc,
        inputs=inputs.reshape([-1, model.params.input_size]),
        labels=labels.reshape([-1, model.params.output_size]),
        states=states.squeeze(),
        pred_labels=pred_labels.reshape([-1, model.params.output_size]))

    for hyper in hyper_params:
        if valid_second_level_params(hyper):
            model.params.update(hyper)
            level2_outdir = utils.create_path(outdir, hyper)
            optimize_wout(
                outdir=level2_outdir, model=model, inputs=inputs,
                states=states, train_labels=labels, pred_labels=pred_labels,
                tikhonov_betas=tikhonov_betas)
        else:
            raise ValueError(
                f"Not all hyper_params are of second level: {hyper_params}")


def optimize_wout(outdir, model, inputs, states, train_labels, pred_labels, tikhonov_betas):
    """Run optimization of last ESN layer (Wout) both with Pseudo Inverse and
    Tikhonov with a list of regularization parameter values."""
    if not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = model.params.transient_length
    plen = model.params.pred_length
    init_input = train_labels[-1]
    init_state = states[-1]

    def _optimize(model, method, beta):
        model.params.train_method = method
        model.params.tikhonov_beta = beta
        logger.info(f"Optimizing Wout with method:{method} / beta:{beta}")
        model.optimize(
            inputs=inputs[tlen:], states=states[tlen:], labels=train_labels[tlen:])

        logger.info(f"Predicting {plen} steps")
        outputs, out_states = model.predict(
            init_input, init_state, nr_predictions=plen)

        if method == "pinv":
            method_outdir = outdir / "pinv"
        elif method == "tikhonov":
            method_outdir = outdir / f"tik{beta}"

        pred_data_nc = method_outdir / "pred_data.nc"
        logger.info(f"Dumping prediction data at {pred_data_nc}")
        utils.dump_prediction(
            fname=pred_data_nc,
            outputs=outputs.reshape([-1, model.params.output_size]),
            labels=pred_labels.reshape([-1, model.params.output_size]),
            states=out_states.squeeze())

        logger.info(f"Dumping model at {method_outdir}/[params.json, model.pth]")
        utils.save_model(method_outdir, model)

    _optimize(model, method="pinv", beta=None)

    for beta in tikhonov_betas:
        _optimize(model, method="tikhonov", beta=beta)


def _tikhonov_optimize(tikhonov_betas, model, params, inputs, states, labels, pred_labels):
    """Find optimal regularization parameter beta out of a given list of tikhonov_betas.
    The generated states would be always the same for varying beta, so this can be
    run independently of the other hyper-parameter searches.
    """
    tlen = params.transient_length

    metrics = []
    for beta in tikhonov_betas:
        model.optimize(
            inputs=inputs[tlen:],
            states=states[tlen:],
            labels=labels[tlen:],
            method="tikhonov",
            beta=beta)

        init_inputs = labels[-1]
        outputs, _ = model.predict(
            init_inputs, states[-1], nr_predictions=params.pred_length)

        error = (outputs - pred_labels)**2
        metric = torch.mean(error).item()
        if not np.isfinite(metric):
            metric = 1e7
        metrics.append(metric)
    min_tik = np.argmin(metrics)
    return tikhonov_betas[min_tik], metrics[min_tik]


def esn_tikhonov_fitnessfunc(loader, params, dimensions, tikhonov_betas, nr_avg_runs=10):
    """Fitness function for hyper-parameter optimization that automatically
    uses the best regularization parameter out of a given list of tikhonov_betas

    Parameters
    ----------
    loader : torch.utils.DataLoader
        loads the necessary inputs/labels/pred_labels
    params : torsk.Params
        model/training parameters
    dimensions : list of skopt.Dimesion
        defining the hyper-parameter search space
    tikhonov_betas : list
        contains the tikhonov regularization parameters that will be tested

    Returns
    -------
    fitness : function
        function to be passed to skopt.gp_minimize
    """
    @use_named_args(dimensions=dimensions)
    def fitness(**sampled_params):

        params.update(sampled_params)

        metrics, betas = [], []
        for _ in tqdm(range(nr_avg_runs)):
            model = ESN(params)

            # create states
            model.eval()  # because we are not using gradients
            inputs, labels, pred_labels, orig_data = next(loader)
            zero_state = torch.zeros(1, params.hidden_size)
            _, states = model(inputs, zero_state, states_only=True)

            # find best tikhonov beta
            tikhonov_beta, metric = _tikhonov_optimize(
                tikhonov_betas=tikhonov_betas, model=model, params=params,
                inputs=inputs, states=states, labels=labels, pred_labels=pred_labels)

            params.tikhonov_beta = tikhonov_beta
            metrics.append(metric)
            betas.append(tikhonov_beta)
        logger.info("Tested parameters:")
        for key, val in sampled_params.items():
            logger.info(f"{key}: {val}")
        logger.info(f"Best tikhonov: {np.median(tikhonov_beta)}")
        return np.median(metric)
    return fitness
