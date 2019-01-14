import numpy as np
import torsk
from torsk.data.numpy_dataset import NumpyImageDataset


def test_image_dataset():
    params = torsk.Params(params={
        "input_shape": [30, 30],
        "input_map_specs": [
            {"type": "pixels", "size": [10, 10], "input_scale": 3}],

        "reservoir_representation": "dense",
        "spectral_radius": 2.0,
        "density": 1e-1,

        "train_length": 10,
        "pred_length": 5,
        "transient_length": 5,
        "train_method": "pinv_svd",

        "dtype": "float64",
        "backend": "numpy",
        "debug": False
    })

    # test dtypes
    for dtype_str in ["float32", "float64"]:
        params.dtype = dtype_str
        dtype = np.dtype(params.dtype)
        height = params.input_shape[0]
        width = params.input_shape[1]

        images = np.zeros(
            [params.pred_length + params.train_length + 2, height, width],
            dtype=params.dtype)
        images[:, 1:3, 1:3] = 2.
        images[:, 0, 0] = -0.3

        dataset = NumpyImageDataset(images, params)
        assert len(dataset) == 2

        inputs, labels, pred_labels = dataset[1]

        assert inputs.shape == (params.train_length,) + images.shape[1:]
        assert labels.shape == (params.train_length,) + images.shape[1:]
        assert pred_labels.shape == (params.pred_length,) + images.shape[1:]

        for arr in dataset[0]:
            assert np.all(arr <= 1.)
            assert np.all(arr >= -1.)
            assert arr.dtype == dtype

        unscaled = dataset.unscale(inputs)
        assert np.isclose(unscaled.max(), 2.)
        assert np.isclose(unscaled.min(), -0.3)
