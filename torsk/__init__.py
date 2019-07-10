# coding: future_fstrings
import time
import logging
import pathlib

import joblib
import numpy as np
import bohrium as bh
import netCDF4 as nc

from torsk.params import Params, default_params
from torsk.imed import imed_metric, eucd_metric
from torsk.numpy_accelerate import to_bh, to_np
from torsk.numpy_accelerate import before_storage, after_storage, numpyize

__all__ = ["Params", "default_params", "load_model", "save_model"]

logger = logging.getLogger(__name__)


def _save_numpy_model(model_pth, model, prefix):
    old_state = before_storage(model)    
    joblib.dump(model, model_pth)
    after_storage(model,old_state)


def _load_numpy_model(model_pth):
    from torsk.models.numpy_esn import NumpyESN
    loaded = joblib.load(model_pth)
    model = NumpyESN(loaded.params)
    numpyize(model)
    if model.params.reservoir_representation == "dense":
        model.esn_cell.weight_hh[:] = loaded.esn_cell.weight_hh[:]
    else:
        model.esn_cell.weight_hh.values[:] = loaded.esn_cell.weight_hh.values[:]
        model.esn_cell.weight_hh.col_idx[:] = loaded.esn_cell.weight_hh.col_idx[:]
    after_storage(model)
    return model

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
    if model.params.backend == "bohrium":
        modelfile = modeldir / f"{prefix}model.pkl"
        
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
        after_storage(model)
    elif params.backend == "torch":
        model_pth = modeldir / f"{prefix}model.pth"
        model = _load_torch_model(model_pth, params)


    return model


def initial_state(hidden_size, dtype, backend):
    if backend == "numpy":
        zero_state = bh.zeros([hidden_size], dtype=np.float64)
    elif backend == "torch":
        import torch
        zero_state = torch.zeros([1, hidden_size], dtype=dtype)
    else:
        raise ValueError("Unkown backend: {backend}")
    return zero_state


def dump_cycles(dst, dataset):
    dst.createDimension("cycle_length", dataset.params.cycle_length)
    dst.createDimension("three", 3)
    dst.createVariable(
        "quadratic_trend", float, ["image_height", "image_width", "three"])
    dst.createVariable(
        "mean_cycle", float, ["image_height", "image_width", "cycle_length"])

    dst["quadratic_trend"][:] = dataset.quadratic_trend
    dst["mean_cycle"][:] = dataset.mean_cycle

    dst.setncatts({
        "cycle_timescale": dataset.cycle_timescale,
        "cycle_length": dataset.params.cycle_length
    })


def dump_training(fname, dataset, idx, states, attrs=None):
    inputs, labels, pred_labels = dataset[idx]

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

        dst.createVariable("inputs", float, ["train_length", "image_height", "image_width"])
        dst.createVariable("labels", float, ["train_length", "image_height", "image_width"])
        dst.createVariable("states", float, ["train_length", "hidden_size"])
        dst.createVariable("pred_labels", float, ["pred_length", "image_height", "image_width"])

        if "cycle_length" in dataset.params.dict:
            dump_cycles(dst, dataset)

        if attrs is not None:
            dst.setncatts(attrs)

        dst["inputs"][:] = to_np(inputs)
        dst["labels"][:] = to_np(labels)
        dst["states"][:] = to_np(states)
        dst["pred_labels"][:] = to_np(pred_labels)


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

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("pred_length", outputs.shape[0])
        dst.createDimension("image_height", outputs.shape[1])
        dst.createDimension("image_width", outputs.shape[2])
        dst.createDimension("hidden_size", states.shape[1])

        dst.createVariable(
            "outputs", float, ["pred_length", "image_height", "image_width"])
        dst.createVariable(
            "labels", float, ["pred_length", "image_height", "image_width"])
        dst.createVariable("states", float, ["pred_length", "hidden_size"])
        dst.createVariable("imed", float, ["pred_length"])
        dst.createVariable("eucd", float, ["pred_length"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["outputs"][:] = outputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["imed"][:] = imed_metric(outputs, labels)
        dst["eucd"][:] = eucd_metric(outputs, labels)


def train_esn(model, dataset, outdir):
    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    inputs, labels, _ = dataset[0]

    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    zero_state = initial_state(hidden_size, dtype, backend)
    _, states = model.forward(inputs, zero_state, states_only=True)
    if outdir is not None:
        outfile = outdir / f"train_data_idx0.nc"
        logger.info(f"Saving training to {outfile}")
        dump_training(outfile, dataset, 0, states=states)

    logger.info("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    save_model(outdir, model)
    return inputs, states, labels



def train_predict_esn(model, dataset, outdir=None, shuffle=False, steps=1,
                      step_length=1, step_start=0):
    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = model.params.transient_length
    hidden_size = model.esn_cell.hidden_size
    backend = model.params.backend
    dtype = model.esn_cell.dtype

    if shuffle:
        logger.warning("If shuffle is True `step_length` has no effect!")

    for ii in range(steps):
        model.timer.reset()
        
        logger.info(f"--- Train/Predict Step Nr. {ii+1} ---")
        if shuffle:
            idx = np.random.randint(low=0, high=len(dataset))
        else:
            idx = ii * step_length + step_start
        inputs, labels, pred_labels = dataset[idx]

        logger.info(f"Creating {inputs.shape[0]} training states")
        zero_state = initial_state(hidden_size, dtype, backend)
        t1 = time.time()
        _, states = model.forward(inputs, zero_state, states_only=True)
        t2 = time.time()
        logger.info(f"Creating {inputs.shape[0]} training states took: {t2-t1}")

        if outdir is not None:
            outfile = outdir / f"train_data_idx{idx}.nc"
            logger.info(f"Saving training to {outfile}")
            dump_training(outfile, dataset, idx, states=states)

        logger.info("Optimizing output weights")
        t1 = time.time()
        model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])
        t2 = time.time()
        logger.info(f"Optimizing output weights took: {t2-t1}")

        if outdir is not None:
            save_model(outdir, model, prefix=f"idx{idx}")

        logger.info(f"Predicting the next {model.params.pred_length} frames")
        init_inputs = labels[-1]
        t1 = time.time()
        outputs, out_states = model.predict(
            init_inputs, states[-1], nr_predictions=model.params.pred_length)
        t2 = time.time()
        logger.info(f"Predicting the next {model.params.pred_length} frames took: {t2-t1}")

        logger.info(model.timer.pretty_print())        
        
        if outdir is not None:
            outfile = outdir / f"pred_data_idx{idx}.nc"
            logger.info(f"Saving prediction to {outfile}")
            dump_prediction(
                outfile, outputs=to_np(outputs), labels=to_np(pred_labels), states=to_np(out_states))

    logger.info(f"Done")
    return model, outputs, pred_labels
