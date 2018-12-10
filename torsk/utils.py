import logging
import pathlib
import numpy as np
import netCDF4 as nc
import torch

import torsk
from torsk.models import ESN

logger = logging.getLogger(__name__)


def create_training_states(model, inputs):
    zero_state = torch.zeros(1, model.params.hidden_size)
    _, states = model(inputs, zero_state, states_only=True)
    return states


def save_model(modeldir, model):
    if not isinstance(modeldir, pathlib.Path):
        modeldir = pathlib.Path(modeldir)
    if not modeldir.exists():
        modeldir.mkdir(parents=True)

    model_pth = modeldir / "model.pth"
    params_json = modeldir / "params.json"
    state_dict = model.state_dict()

    # convert sparse tensor
    key = "esn_cell.weight_hh"
    if isinstance(state_dict[key], torch.sparse.FloatTensor):
        # TODO: can be removed when save/load is implemented for sparse tensors
        # discussion: https://github.com/pytorch/pytorch/issues/9674
        weight = state_dict.pop(key)
        state_dict[key + "_indices"] = weight.coalesce().indices()
        state_dict[key + "_values"] = weight.coalesce().values()

    model.params.save(params_json.as_posix())
    torch.save(state_dict, model_pth.as_posix())


def load_model(modeldir):
    if isinstance(modeldir, str):
        modeldir = pathlib.Path(modeldir)

    params = torsk.Params(modeldir / "params.json")
    model = ESN(params)
    state_dict = torch.load(modeldir / "model.pth")

    # restore sparse tensor
    key = "esn_cell.weight_hh"
    key_idx = key + "_indices"
    key_val = key + "_values"
    if key_idx in state_dict:
        # TODO: can be removed when save/load is implemented for sparse tensors
        # discussion: https://github.com/pytorch/pytorch/issues/9674
        weight_idx = state_dict.pop(key_idx)
        weight_val = state_dict.pop(key_val)
        hidden_size = params.hidden_size
        weight_hh = torch.sparse.FloatTensor(
            weight_idx, weight_val, [hidden_size, hidden_size])
        state_dict[key] = weight_hh

    model.load_state_dict(state_dict)

    return model


def dump_training(fname, inputs, labels, states, pred_labels, attrs=None):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.numpy()

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("train_length", inputs.shape[0])
        dst.createDimension("pred_length", pred_labels.shape[0])
        dst.createDimension("inputs_size", inputs.shape[1])
        dst.createDimension("outputs_size", labels.shape[1])
        dst.createDimension("hidden_size", states.shape[1])

        dst.createVariable("inputs", float, ["train_length", "inputs_size"])
        dst.createVariable("labels", float, ["train_length", "outputs_size"])
        dst.createVariable("states", float, ["train_length", "hidden_size"])
        dst.createVariable("pred_labels", float, ["pred_length", "outputs_size"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["inputs"][:] = inputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["pred_labels"][:] = pred_labels


def dump_prediction(fname, outputs, labels, states, attrs=None):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(states, torch.Tensor):
        states = states.numpy()

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    error = (outputs - labels)**2
    rmse = np.mean(error)**.5

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("pred_length", outputs.shape[0])
        dst.createDimension("output_size", outputs.shape[1])
        dst.createDimension("hidden_size", states.shape[1])
        dst.createDimension("scalar", 1)

        dst.createVariable("outputs", float, ["pred_length", "output_size"])
        dst.createVariable("labels", float, ["pred_length", "output_size"])
        dst.createVariable("states", float, ["pred_length", "hidden_size"])
        dst.createVariable("rmse", float, ["scalar"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["outputs"][:] = outputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["rmse"][:] = rmse


def create_path(root, param_dict, prefix='level2', postfix=None):
    if not isinstance(root, pathlib.Path):
        root = pathlib.Path(root)
    folder = prefix
    for key, val in param_dict.items():
        folder += f"-{key}:{val}"
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
