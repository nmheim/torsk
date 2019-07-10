import pathlib
import logging
import numpy as np

import torsk
# from torsk.imed import imed_metric
from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
from torsk.models.numpy_esn import NumpyESN as ESN

def test_kuro():
    np.random.seed(0)

    params = torsk.default_params()
    params.input_map_specs = [
        {"type": "pixels", "size": [30, 30], "input_scale": 3.},
        {"type": "conv", "mode": "same", "size": [5, 5],
            "kernel_type":"gauss", "input_scale": 2.},
        {"type": "conv", "mode": "same", "size": [10,10],
            "kernel_type":"gauss", "input_scale": 1.5},
        {"type": "conv", "mode": "same", "size": [15, 15],
            "kernel_type":"gauss", "input_scale": 1.},
        {"type": "conv", "mode": "same", "size": [ 5, 5],
            "kernel_type":"random", "input_scale": 1.},
        {"type": "conv", "mode": "same", "size": [10, 10],
            "kernel_type":"random", "input_scale": 1.},
        {"type": "conv", "mode": "same", "size": [20, 20],
            "kernel_type":"random", "input_scale": 1.},
        {"type": "dct", "size": [15, 15], "input_scale": 1.},
        {"type": "gradient", "input_scale": 1.},
        {"type": "gradient", "input_scale": 1.}
    ]

    params.spectral_radius = 1.5
    params.density = 0.01
    params.input_shape = [30, 30]
    params.train_length = 12*73
    params.pred_length = 73
    params.transient_length = 3*73
    params.dtype = "float64"
    params.reservoir_representation = "sparse"
    params.backend = "numpy"
    params.train_method = "pinv_lstsq"
    params.tikhonov_beta = 3e1
    params.debug = False
    params.imed_loss = True
    params.imed_sigma = 1.0

    logger = logging.getLogger(__file__)
    level = "DEBUG" if params.debug else "INFO"
    logging.basicConfig(level=level)
    logging.getLogger("matplotlib").setLevel("INFO")

    images = np.load(pathlib.Path(__file__).parent / "kuro_test_sequence.npy")
    dataset = ImageDataset(images, params, scale_images=True)

    logger.info("Building model ...")
    model = ESN(params)

    logger.info("Training + predicting ...")
    model, outputs, pred_labels = torsk.train_predict_esn(model, dataset)

    # logger.info("Visualizing results ...")
    # import matplotlib.pyplot as plt
    # from torsk.visualize import animate_double_imshow
    # anim = animate_double_imshow(pred_labels, outputs)
    # plt.show()

    error = np.abs(outputs - pred_labels)
    logger.info(error.mean())
    logger.info(error.max())
    assert error.mean() < 0.2
    assert error.max() < 1.3
