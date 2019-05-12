import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torsk
from torsk.models.torch_lstm import LSTM
from torsk.visualize import animate_double_imshow
from torsk.data.numpy_dataset import NumpyImageDataset
from train import circle_train_eval_test, get_params


def esn_perf(hidden_size):
    output_dir = pathlib.Path(f"esn_output_{hidden_size}")
    model = torsk.load_model(output_dir)
    params = torsk.Params(output_dir / "params.json")
    params.train_length = 100
    params.transient_length = 100
    params.pred_length = 300
    params.batch_size = 32
    
    _, _, test = circle_train_eval_test(params.input_shape)
    dataset = NumpyImageDataset(test[:450], params)
    inputs, _, _ = dataset[0]
    
    esn_error_path = output_dir / "esn_error.npy"
    if not esn_error_path.exists():
        esn_error = []
        print("Generating ESN predictions")
        for inputs, labels, pred_labels in tqdm(dataset):
            zero_state = np.zeros(model.esn_cell.hidden_size)
            _, states = model.forward(inputs, zero_state, states_only=True)
            pred, _ = model.predict(
                labels[-1], states[-1],
                nr_predictions=params.pred_length)
            err = np.abs(pred - pred_labels)
            esn_error.append(err.squeeze())
        
        esn_error = np.mean(esn_error, axis=0)
        np.save(esn_error_path, esn_error)
    else:
        esn_error = np.load(esn_error_path)

    return esn_error
 
def lstm_perf(hidden_size):
    hp = get_params()
    hp.hidden_size = hidden_size
    output_dir = f"lstm_output_h{hp.hidden_size}"
    input_size = hp.input_shape[0] * hp.input_shape[1]
    lstm_model = LSTM(input_size, hp.hidden_size)
    lstm_model.load_state_dict(torch.load(f"{output_dir}/lstm_model_6.pth"))
    
    _, _, loader = get_data_loaders(hp)
    inputs, labels, pred_labels = next(iter(loader))
    
    print("Generating LSTM predictions")
    lstm_pred = lstm_model.predict(inputs, steps=hp.pred_length)
    lstm_pred = lstm_pred.detach().squeeze().numpy()
    pred_labels = pred_labels.detach().squeeze().numpy()
    lstm_error = np.abs(lstm_pred - pred_labels).mean(axis=0)
    
    example_pred = lstm_pred[0].reshape(hp.pred_length, *hp.input_shape)
    example_lbls = pred_labels[0].reshape(hp.pred_length, *hp.input_shape)
    anim = animate_double_imshow(example_lbls, example_pred)
    plt.show()

hidden_size = 4096
lstm_perf(hidden_size)
