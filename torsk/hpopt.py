import logging
import numpy as np
from skopt.utils import use_named_args
import torch
from tqdm import tqdm
from torsk.models import ESN


logger = logging.getLogger(__name__)


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
