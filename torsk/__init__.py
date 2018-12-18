import pathlib
import logging
import numpy as np

from torsk.params import Params, default_params

logger = logging.getLogger(__name__)
__all__ = ["Params", "default_params"]


def train_predict_esn(model, dataset, outdir=None, shuffle=True):

    if outdir is not None and not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir)

    tlen = model.params.transient_length
    # model.eval()  # because we are not using gradients

    ii = np.random.randint(low=0, high=len(dataset)) if shuffle else 0
    inputs, labels, pred_labels, orig_data = dataset[ii]

    logger.debug(f"Creating {inputs.shape[0]} training states")
    zero_state = np.zeros([model.esn_cell.hidden_size], dtype=model.esn_cell.dtype)
    _, states = model.forward(inputs, zero_state, states_only=True)

    logger.debug("Optimizing output weights")
    model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    logger.debug(f"Predicting the next {model.params.pred_length} frames")
    init_inputs = labels[-1]
    outputs, out_states = model.predict(
        init_inputs, states[-1], nr_predictions=model.params.pred_length)

    # if outdir is not None:
    #     outfile = outdir / "train_data.nc"
    #     logger.debug(f"Saving training to {outfile}")
    #     dump_training(
    #         outfile,
    #         inputs=inputs.reshape([-1, params.input_size]),
    #         labels=labels.reshape([-1, params.output_size]),
    #         states=states.reshape([-1, params.hidden_size]),
    #         pred_labels=pred_labels.reshape([-1, params.output_size]))

    # logger.debug("Optimizing output weights")
    # model.optimize(inputs=inputs[tlen:], states=states[tlen:], labels=labels[tlen:])

    # if outdir is not None:
    #     modelfile = outdir / "model.pth"
    #     logger.debug(f"Saving model to {modelfile}")
    #     save_model(modelfile.parent, model)

    # logger.debug(f"Predicting the next {params.pred_length} frames")
    # init_inputs = labels[-1]
    # outputs, out_states = model.predict(
    #     init_inputs, states[-1], nr_predictions=model.params.pred_length)

    # if outdir is not None:
    #     outfile = outdir / "pred_data.nc"
    #     logger.debug(f"Saving prediction to {outfile}")
    #     dump_prediction(outfile,
    #         outputs=outputs.reshape([-1, params.input_size]),
    #         labels=pred_labels.reshape([-1, params.output_size]),
    #         states=out_states.reshape([-1, params.hidden_size]))

    return model, outputs, pred_labels

# from torsk.utils import dump_training, dump_prediction, save_model, load_model
