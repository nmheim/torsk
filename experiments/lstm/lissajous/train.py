import time
import pathlib
import warnings
import logging
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
from torsk.data.utils import mackey_sequence, gauss2d_sequence
from torsk.data.torch_dataset import TorchImageDataset
from torsk.data.numpy_dataset import NumpyImageDataset

logging.basicConfig(level="INFO")


def circle_train_eval_test(input_shape):
    t = np.arange(0, 200*np.pi, 0.1)[:4000]
    x, y = np.sin(0.3 * t), np.cos(t)
    center = np.array([y, x]).T
    images = gauss2d_sequence(center, sigma=0.5, size=input_shape)

    train, val, test = images[:2500], images[2500:3000], images[3000:]
    return train, val, test

def get_data_loaders(hp):
    train, val, test = circle_train_eval_test(hp.input_shape)
    
    train_ds = TorchImageDataset(train, hp)
    eval_ds = TorchImageDataset(val, hp)
    test_ds = TorchImageDataset(test, hp, return_pred_labels=True)
    train_dl = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=1)
    eval_dl = DataLoader(
        eval_ds, batch_size=val.shape[0], shuffle=True, num_workers=1)
    test_dl = DataLoader(
        test_ds, batch_size=test.shape[0], shuffle=True, num_workers=1)
    return train_dl, eval_dl, test_dl

def get_params():
    hp = torsk.Params()
    hp.dtype = "float32"
    hp.train_length = 200
    hp.pred_length = 100
    hp.batch_size = 32
    hp.hidden_size = 4096
    hp.input_shape = [10, 10]
    return hp


def run_lstm():

    hp = get_params()
    output_dir = f"lstm_output_h{hp.hidden_size}"
    
    train_dl, eval_dl, test_dl = get_data_loaders(hp)
    input_size = hp.input_shape[0] * hp.input_shape[1]
    model = LSTM(input_size=input_size, hidden_size=hp.hidden_size)
    params = model.parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss = nn.MSELoss()
    metrics = {'loss': Loss(loss)}
    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model, metrics=metrics)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "lstm", save_interval=1, n_saved=10, require_empty=False)
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
    
    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, {"model":model})
        else:
            raise e
    
    trainer.run(train_dl, max_epochs=20)

def run_esn():
    params = get_params()
    output_dir = f"esn_output_h{params.hidden_size}"
    params.input_map_specs = [
        {"type": "random_weights", "size": [params.hidden_size], "input_scale": 1.}
    ]
    params.spectral_radius = 1.5
    params.density = 0.1
    params.train_length = 2200
    params.transient_length = 200
    params.dtype = "float64"
    params.reservoir_representation = "dense"
    params.backend = "numpy"
    params.train_method = "pinv_svd"
    params.tikhonov_beta = 2.0
    params.debug = False
    params.imed_loss = False

    output_dir = pathlib.Path(f"esn_output_{params.hidden_size}")
    model = ESN(params)
    model_path = output_dir / "model.pkl"
    
    train, _, test = circle_train_eval_test(params.input_shape)
    dataset = NumpyImageDataset(train, params)
    
    t1 = time.time()
    torsk.train_esn(model, dataset, outdir=output_dir)
    t2 = time.time()
    print(f"ESN Training Time: {t2-t1} s")
    
if __name__ == "__main__":
    run_lstm()
    run_esn()
