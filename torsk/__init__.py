import json
import logging
import torch
import netCDF4 as nc

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


def train_predict_esn(model, loader, params, outfile=None, modelfile=None):

    model.eval()  # because we are not using gradients
    tlen = params.transient_length

    inputs, labels, pred_labels, orig_data = next(loader)

    logger.debug(f"Creating {inputs.size(0)} training states")
    zero_state = torch.zeros(1, params.hidden_size)
    _, states = model(inputs, zero_state, states_only=True)

    if outfile is not None:
        logger.debug(f"Saving states to {outfile}")
        dump_states(outfile, states.squeeze().numpy())

    logger.debug("Optimizing output weights")
    model.optimize(
        inputs=inputs[tlen:],
        states=states[tlen:],
        labels=labels[tlen:],
        method=params.train_method,
        beta=params.tikhonov_beta)
    if modelfile is not None:
        logger.debug(f"Saving model to {modelfile}")
        save_model(model, modelfile)

    logger.debug(f"Predicting the next {params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, _ = model.predict(
        init_inputs, states[-1], nr_predictions=params.pred_length)

    if outfile is not None:
        logger.debug(f"Saving outputs to {outfile}")
        if hasattr(loader.dataset, "to_image"):
            np_outputs = loader.dataset.to_image(outputs)
            np_labels = loader.dataset.to_image(pred_labels)
        else:
            np_outputs = outputs.numpy()
            np_labels = pred_labels.numpy()

        dump_outputs(
            fname=outfile, outputs=np_outputs, labels=np_labels, mode="a")

    return model, outputs, pred_labels, orig_data
