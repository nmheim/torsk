import time
import pathlib
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, Timer

import torsk
from torsk.models.torch_lstm import LSTM
from torsk.models.numpy_esn import NumpyESN as ESN
from torsk.data.numpy_dataset import NumpyImageDataset
from torsk.data.utils import mackey_sequence
from torsk.data.torch_dataset import TorchImageDataset
logging.basicConfig(level="INFO")


def esn_params(hidden_size):
    params = torsk.Params()
    params.input_map_specs = [
        {"type": "random_weights", "size": [hidden_size], "input_scale": 1.}
    ]
    params.spectral_radius = 1.5
    params.density = 0.1
    params.input_shape = [1, 1]
    params.train_length = 2200
    params.pred_length = 300
    params.transient_length = 200
    params.dtype = "float64"
    params.reservoir_representation = "dense"
    params.backend = "numpy"
    params.train_method = "pinv_svd"
    params.tikhonov_beta = 2.0
    params.debug = False
    params.imed_loss = False
    return params


def mackey_train_eval_test():
    mackey = mackey_sequence(N=5000)
    mackey_train, mackey_eval = mackey[:3000], mackey[3000:4000]
    mackey_test = mackey[4000:]
    return mackey_train, mackey_eval, mackey_test


def get_data_loaders(hp):
    mackey_train, mackey_eval, mackey_test = mackey_train_eval_test()
    train_ds = TorchImageDataset(mackey_train[:, None, None], hp)
    eval_ds = TorchImageDataset(mackey_eval[:, None, None], hp)
    test_ds = TorchImageDataset(mackey_test[:, None, None], hp, return_pred_labels=True)
    train_dl = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=1)
    eval_dl = DataLoader(
        eval_ds, batch_size=mackey_eval.shape[0], shuffle=True, num_workers=1)
    test_dl = DataLoader(
        test_ds, batch_size=mackey_test.shape[0], shuffle=True, num_workers=1)
    return train_dl, eval_dl, test_dl


def lstm_params(hidden_size):
    hp = torsk.Params()
    hp.dtype = "float32"
    hp.train_length = 200
    hp.pred_length = 300
    hp.batch_size = 32
    hp.hidden_size = hidden_size
    return hp


def run_lstm(outdir, hidden_size):
    hp = lstm_params(hidden_size)
    train_dl, eval_dl, test_dl = get_data_loaders(hp)
    model = LSTM(input_size=1, hidden_size=hp.hidden_size)
    params = model.parameters()
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss = nn.MSELoss()
    metrics = {'loss': Loss(loss)}
    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model, metrics=metrics)
    checkpoint_handler = ModelCheckpoint(
        outdir, "lstm", save_interval=1, n_saved=10, require_empty=False)
    timer = Timer(average=True)
    
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
        to_save={"model":model})
    
    timer.attach(
        trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)
    
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.4f}".format(trainer.state.epoch, trainer.state.output))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_dl)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg loss: {:.4f}"
              .format(trainer.state.epoch, metrics['loss']))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(eval_dl)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg loss: {:.4f}"
              .format(trainer.state.epoch, metrics['loss']))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_times(trainer):
        print(f"Time per epoch: {timer.value()}")
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(trainer):
        fig, ax = plt.subplots(1,1)
        inputs, labels, pred_labels = next(iter(test_dl))
        pred = model.predict(inputs, steps=100)
        pred = pred.detach().squeeze().numpy()
        pred_labels = pred_labels.detach().squeeze().numpy()
        ax.plot(pred)
        ax.plot(pred_labels)
        fig.savefig(f"{outdir}/pred_{trainer.state.epoch}.pdf")
        plt.close()
    
    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            create_plots(engine)
            checkpoint_handler(engine, {"model":model})
        else:
            raise e
    
    trainer.run(train_dl, max_epochs=20)


def run_online_esn(outdir, hidden_size):
    params = esn_params(hidden_size)
    model = ESN(params)
    mackey_train, mackey_eval, mackey_test = mackey_train_eval_test()
    dataset = NumpyImageDataset(mackey_train[:, None, None], params)
    model, outputs, pred_labels = torsk.train_predict_esn(
        model, dataset, outdir,
        steps=100, step_length=5, step_start=0)

def run_offline_esn(outdir, hidden_size):
    outdir = pathlib.Path(outdir)
    params = esn_params(hidden_size)
    model = ESN(params)
    mackey_train, mackey_eval, mackey_test = mackey_train_eval_test()
    dataset = NumpyImageDataset(mackey_train[:, None, None], params)

    t1 = time.time()
    torsk.train_esn(model, dataset, outdir=outdir)
    t2 = time.time()
    print(f"ESN Training Time: {t2-t1} s")


if __name__ == "__main__":
    outdir = pathlib.Path("/mnt/data/torsk_experiments/mackey_1d_esn_lstm")
    offline_esn_dir = outdir / "offline_esn"
    online_esn_dir = outdir / "online_esn"
    lstm_dir = outdir / "lstm"

    hidden_size = 1000

    run_lstm(lstm_dir, hidden_size)
    run_offline_esn(offline_esn_dir, hidden_size)
    run_online_esn(online_esn_dir, hidden_size)
