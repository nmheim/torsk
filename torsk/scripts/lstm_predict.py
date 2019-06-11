import pathlib
import click


@click.command("lstm-predict", short_help="Find anomalies in multiple predicition runs")
@click.argument("lstm_path", type=pathlib.Path)
@click.argument("train_data_ncfiles", nargs=-1, type=pathlib.Path)
@click.option("--pred-length", "-p", type=int, default=100,
    help="Prediction length for lstm-based prediction")
@click.option("--train-length", "-t", type=int, default=730,
    help="Train length for lstm-based prediction")
@click.option("--params", type=pathlib.Path, default=None)
def cli(lstm_path, train_data_ncfiles, pred_length, train_length, params=None):
    """Create lstm-based prediction. Results are stored in
    `lstm_{train_data_nc.stem}.npy` and overwritten if already present.
    """
    import torch
    from tqdm import tqdm
    import numpy as np
    import netCDF4 as nc

    import torsk
    from torsk.scripts.pred_perf import sort_filenames
    from torsk.models.torch_lstm import LSTM

    if params is None:
        params = torsk.Params()
        params.dtype = "float32"
        params.train_length = 200
        params.pred_length = 300
        params.batch_size = 32
        params.hidden_size = 10000
        params.input_shape = [30, 30]
    else:
        params = torsk.Params(json_path=params)

    input_size = params.input_shape[0] * params.input_shape[1]
    lstm_model = LSTM(input_size, params.hidden_size)
    lstm_model.load_state_dict(torch.load(lstm_path))

    train_data_ncfiles = sort_filenames(train_data_ncfiles)

    for train_data_nc in tqdm(train_data_ncfiles):
        assert "train" in train_data_nc.as_posix()
        with nc.Dataset(train_data_nc, "r") as src:
            labels = src["labels"][-train_length:]
            labels = torch.Tensor(labels.reshape([1, train_length, -1]))
            lstm_pred = lstm_model.predict(labels, steps=pred_length)
            lstm_pred = lstm_pred.detach().numpy().reshape([pred_length,] + params.input_shape)
            name = train_data_nc.stem.replace("train", "pred")
            outfile = train_data_nc.parent / f"lstm_{name}.npy"
            tqdm.write(f"Saving lstm-based prediction at {outfile}")
            np.save(outfile, lstm_pred)
