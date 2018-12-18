import numpy as np
import torsk
from torsk.data.image import NumpyImageDataset, gauss2d


def test_numpy_dataset():

    params = torsk.default_params()
    params.feature_specs[0]["xsize"] = [3, 3]
    params.feature_specs[1]["ksize"] = [3, 3]
    params.feature_specs[2]["kernel_shape"] = [2, 2]

    images = np.zeros(
        [params.pred_length + params.train_length + 2, 4, 4], dtype=params.dtype)
    images[:, 1:3, 1:3] = 1.

    dataset = NumpyImageDataset(images, params)
    assert len(dataset) == 2

    features = dataset.to_features(images)
    assert features.shape == (images.shape[0], 9 + 9 + 9)
