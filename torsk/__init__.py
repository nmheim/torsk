import logging
import pathlib

import joblib
import numpy as np
import netCDF4 as nc

from torsk.params import Params, default_params

__all__ = ["Params", "default_params", "load_model", "save_model"]

logger = logging.getLogger(__name__)


def _save_numpy_model(model_pth, model, prefix):
    joblib.dump(model, model_pth)


def _load_numpy_model(model_pth):
    return joblib.load(model_pth)


def _save_torch_model(model_pth, model, prefix):
    import torch
    state_dict = model.state_dict()

    # convert sparse tensor
    key = "esn_cell.weight_hh"
    if isinstance(state_dict[key], torch.sparse.FloatTensor):
        # TODO: can be removed when save/load is implemented for sparse tensors
        # discussion: https://github.com/pytorch/pytorch/issues/9674
        weight = state_dict.pop(key)
        state_dict[key + "_indices"] = weight.coalesce().indices()
        state_dict[key + "_values"] = weight.coalesce().values()

    torch.save(state_dict, model_pth.as_posix())


def _fix_prefix(prefix):
    if prefix is not None:
        prefix = prefix.strip("-") + "-"
    else:
        prefix = ""
    return prefix


def _load_torch_model(model_pth, params):
    import torch
    from torsk.models.torch_esn import TorchESN
    model = TorchESN(params)
    state_dict = torch.load(model_pth)

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


def save_model(modeldir, model, prefix=None):
    if not isinstance(modeldir, pathlib.Path):
        modeldir = pathlib.Path(modeldir)
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
    prefix = _fix_prefix(prefix)

    params_json = modeldir / f"{prefix}params.json"
    logger.info(f"Saving model parameters to {params_json}")
    model.params.save(params_json)

    if model.params.backend == "numpy":
        modelfile = modeldir / f"{prefix}model.pkl"
        logger.info(f"Saving model to {modelfile}")
        _save_numpy_model(modelfile, model, prefix)
    elif model.params.backend == "torch":
        modelfile = modeldir / f"{prefix}model.pth"
        logger.info(f"Saving model to {modelfile}")
        _save_torch_model(modelfile, model, prefix)


def load_model(modeldir, prefix=None):
    # TODO: fix circular import
    if isinstance(modeldir, str):
        modeldir = pathlib.Path(modeldir)
    prefix = _fix_prefix(prefix)

    params = Params(modeldir / f"{prefix}params.json")

    if params.backend == "numpy":
        model_pth = modeldir / f"{prefix}model.pkl"
        model = _load_numpy_model(model_pth)
    elif params.backend == "torch":
        model_pth = modeldir / f"{prefix}model.pth"
        model = _load_torch_model(model_pth, params)
    return model


def initial_state(hidden_size, dtype, backend):
    if backend == "numpy":
        zero_state = np.zeros([hidden_size], dtype=dtype)
    elif backend == "torch":
        import torch
        zero_state = torch.zeros([1, hidden_size], dtype=dtype)
    else:
        raise ValueError("Unkown backend: {backend}")
    return zero_state


def dump_training(fname, inputs, labels, states, pred_labels, attrs=None):
    if not isinstance(inputs, np.ndarray):
        raise ValueError("Check that this acutally works...")
        msg = "Inputs are not numpy arrays. " \
              "Assuming Tensors of shape [time, batch, features]"
        logger.debug(msg)
        inputs = inputs.numpy().reshape([-1, inputs.size(2)])
        labels = labels.numpy().reshape([-1, labels.size(2)])
        states = states.numpy().reshape([-1, states.size(2)])
        pred_labels = pred_labels.numpy().reshape([-1, pred_labels.size(2)])

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("train_length", inputs.shape[0])
        dst.createDimension("pred_length", pred_labels.shape[0])
        dst.createDimension("image_height", inputs.shape[1])
        dst.createDimension("image_width", inputs.shape[2])
        dst.createDimension("hidden_size", states.shape[1])

        dst.createVariable(
            "inputs", float, ["train_length", "image_height", "image_width"])
        dst.createVariable(
            "labels", float, ["train_length", "image_height", "image_width"])
        dst.createVariable(
            "states", float, ["train_length", "hidden_size"])
        dst.createVariable(
            "pred_labels", float, ["pred_length", "image_height", "image_width"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["inputs"][:] = inputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["pred_labels"][:] = pred_labels


def dump_prediction(fname, outputs, labels, states, attrs=None):
    if not isinstance(outputs, np.ndarray):
        raise ValueError("Check that this acutally works...")
        msg = "Inputs are not numpy arrays. " \
              "Assuming Tensors of shape [time, batch, features]"
        logger.debug(msg)
        outputs = outputs.numpy().reshape([-1, outputs.size(2)])
        labels = labels.numpy().reshape([-1, labels.size(2)])
        states = states.numpy().reshape([-1, states.size(2)])

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    error = (outputs - labels)**2
    rmse = np.mean(error)**.5

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("pred_length", outputs.shape[0])
        dst.createDimension("image_height", outputs.shape[1])
        dst.createDimension("image_width", outputs.shape[2])
        dst.createDimension("hidden_size", states.shape[1])
        dst.createDimension("scalar", 1)

        dst.createVariable(
            "outputs", float, ["pred_length", "image_height", "image_width"])
        dst.createVariable(
            "labels", float, ["pred_length", "image_height", "image_width"])
        dst.createVariable("states", float, ["pred_length", "hidden_size"])
        dst.createVariable("rmse", float, ["scalar"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["outputs"][:] = outputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["rmse"][:] = rmse


def train_predict_esn(model, dataset, outdir=None, shuffle=True):
    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    ii = np.random.randint(low=0, high=len(dataset)) if shuffle else 0
    inputs, labels, pred_labels = dataset[ii]

    logger.info(f"Creating {inputs.shape[0]} training states")
    zero_state = initial_state(hidden_size, dtype, backend)
    _, states = model.forward(inputs, zero_state, states_only=True)

    if outdir is not None:
        outfile = outdir / "train_data.nc"
        logger.info(f"Saving training to {outfile}")
        dump_training(outfile, inputs=inputs, labels=labels, states=states,
                      pred_labels=pred_labels)

    logger.info("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    if outdir is not None:
        save_model(outdir, model)

    logger.info(f"Predicting the next {model.params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, out_states = model.predict(
        init_inputs, states[-1], nr_predictions=model.params.pred_length)

    if outdir is not None:
        outfile = outdir / "pred_data.nc"
        logger.info(f"Saving prediction to {outfile}")
        dump_prediction(
            outfile, outputs=outputs, labels=pred_labels, states=out_states)

    logger.info(f"Done")
    return model, outputs, pred_labels
