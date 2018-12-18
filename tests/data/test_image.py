import numpy as np
import torsk
from torsk.data.numpy_datasets import NumpyImageDataset


def test_image_dataset_shape():

    params = torsk.default_params()
    params.dtype = "float32"
    params.feature_specs = [
        {"type": "pixels", "size": [3, 3]},
        {"type": "dct", "size": [3, 3]},
        {"type": "conv", "size": [15, 15], "kernel_type": "gauss"}
    ]

    images = np.zeros(
        [params.pred_length + params.train_length + 2, 30, 30], dtype=params.dtype)
    images[:, 1:3, 1:3] = 1.

    dataset = NumpyImageDataset(images, params)
    assert len(dataset) == 2

    features = dataset.to_features(images)
    assert features.dtype == np.float32
    assert features.shape == (images.shape[0], 9 + 9 + 16**2)

    labels, pred_labels = dataset.get_images(index=0)
    assert labels.dtype == np.float32
    assert labels.shape == images[1:params.train_length + 1].shape
    assert np.all(labels == images[1:params.train_length + 1])
