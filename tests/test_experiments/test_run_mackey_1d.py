import numpy as np

import torsk
from torsk.data.utils import mackey_sequence, normalize
from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
from torsk.models.numpy_esn import NumpyESN as ESN

def test_run_1dmackey(tmpdir):

    np.random.seed(0)

    params = torsk.default_params()
    params.input_map_specs = [
        {"type": "random_weights", "size": [1000], "input_scale": 1.}
    ]
    params.spectral_radius = 1.5
    params.density = 0.05
    params.input_shape = [1, 1]
    params.train_length = 2200
    params.pred_length = 400
    params.transient_length = 200
    params.dtype = "float64"
    params.reservoir_representation = "dense"
    params.backend = "numpy"
    params.train_method = "pinv_svd"
    params.tikhonov_beta = 2.0
    params.debug = False
    params.imed_loss = False

    model = ESN(params)

    mackey = mackey_sequence(N=3700)
    mackey = normalize(mackey) * 2 - 1
    mackey = mackey[:, np.newaxis, np.newaxis]
    dataset = ImageDataset(mackey, params, scale_images=False)

    model, outputs, pred_labels = torsk.train_predict_esn(
        model, dataset, tmpdir.mkdir("mackey_output"))

    # import matplotlib.pyplot as plt
    # plt.plot(np.squeeze(outputs))
    # plt.plot(np.squeeze(pred_labels))
    # plt.show()

    error = np.abs(outputs - pred_labels)
    assert error.mean() < 0.07
    assert error.max() < 0.4
