import numpy as np
import torsk
from torsk.data.numpy_dataset import NumpyImageDataset


def test_image_dataset():

    params = torsk.default_params()
    params.feature_specs = [
        {"type": "pixels", "size": [3, 3]},
        {"type": "dct", "size": [3, 3]},
        {"type": "conv", "size": [15, 15], "kernel_type": "gauss"}
    ]

    # test dtypes
    for dtype_str in ["float32", "float64"]:
        params.dtype = dtype_str
        dtype = np.dtype(params.dtype)

        images = np.zeros(
            [params.pred_length + params.train_length + 2, 30, 30], dtype=params.dtype)
        images[:, 1:3, 1:3] = 1.

        dataset = NumpyImageDataset(images, params)
        assert len(dataset) == 2

        features = dataset.to_features(images)
        assert features.dtype == dtype

        small_images = dataset.to_images(features)
        assert small_images.dtype == dtype

        for arr in dataset[0]:
            assert arr.dtype == dtype

        img_seqs = dataset.get_raw_images(index=0)
        for imgs in img_seqs:
            assert imgs.dtype == dtype

    # test shapes and values
    assert small_images.shape == (images.shape[0], 3, 3)
    assert features.shape == (images.shape[0], 9 + 9 + 16**2)

    inputs, labels, pred_labels = img_seqs
    assert inputs.shape == images[:params.train_length].shape
    assert labels.shape == images[1:params.train_length + 1].shape
    assert pred_labels.shape == images[-params.pred_length:].shape

    assert np.all(inputs == images[:params.train_length])
    assert np.all(labels == images[1:params.train_length + 1])
    assert np.all(pred_labels == images[-params.pred_length:])
