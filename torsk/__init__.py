import json
import pathlib
import logging
import torch
import netCDF4 as nc
from torsk.utils import dump_training, dump_prediction, save_model, load_model

logger = logging.getLogger(__name__)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, params):
        """Updates parameters based on a dictionary."""
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def mse(predictions, labels):
    err = (predictions - labels)**2
    return torch.mean(err).item()


def train_predict_esn(model, loader, params, outdir=None):
    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    model.eval()  # because we are not using gradients
    tlen = params.transient_length

    inputs, labels, pred_labels, orig_data = next(loader)

    logger.debug(f"Creating {inputs.size(0)} training states")
    zero_state = torch.zeros(1, model.esn_cell.hidden_size)
    _, states = model(inputs, zero_state, states_only=True)

    if outdir is not None:
        outfile = outdir / "train_data.nc"
        logger.debug(f"Saving training to {outfile}")
        dump_training(
            outfile,
            inputs=inputs.reshape([-1, params.input_size]),
            labels=labels.reshape([-1, params.output_size]),
            states=states.reshape([-1, params.hidden_size]),
            pred_labels=pred_labels.reshape([-1, params.output_size]))

    logger.debug("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])
    if outdir is not None:
        modelfile = outdir / "model.pth"
        logger.debug(f"Saving model to {modelfile}")
        save_model(modelfile.parent, model)

    logger.debug(f"Predicting the next {params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, out_states = model.predict(
        init_inputs, states[-1], nr_predictions=params.pred_length)

    if outdir is not None:
        outfile = outdir / "pred_data.nc"
        logger.debug(f"Saving prediction to {outfile}")
        dump_prediction(outfile,
            outputs=outputs.reshape([-1, params.input_size]),
            labels=pred_labels.reshape([-1, params.output_size]),
            states=out_states.reshape([-1, params.hidden_size]))

    return model, outputs, pred_labels, orig_data
