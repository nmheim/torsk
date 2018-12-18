import logging
import numpy as np
from torsk import utils

logger = logging.getLogger(__name__)


def pixel_features(images, xsize):
    """Resample a sequence of images to xsize and reshape them to a sequence of
    feature vectors.
    """
    seq_len = images.shape[0]
    nr_pxl_features = xsize[0] * xsize[1]
    pxl_features = utils.resample2d(images, xsize)
    return pxl_features.reshape([seq_len, nr_pxl_features])


def dct_features(images, ksize):
    """Apply 2d DCT to images, keep only window of ksize coefficients and
    create feature vector sequence.
    """
    seq_len = images.shape[0]
    nr_dct_features = ksize[0] * ksize[1]
    dct_features = utils.dct2_sequence(images, ksize)
    return dct_features.reshape([seq_len, nr_dct_features])


def conv_features(images, spec):
    """Apply 2d convolution specified by spec and reshape to a sequence of
    feature vecotrs.
    """
    seq_len = images.shape[0]
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    conv_features = utils.conv2d(images, **kwargs)
    print(conv_features.shape)
    kwargs = {k: v for k, v in kwargs.items() if k != "kernel_type"}
    size = utils.conv2d_output_shape(images.shape[1:], **kwargs)
    return conv_features.reshape([seq_len, size[0] * size[1]])


class NumpyImageDataset:

    def __init__(self, images, params):
        self.train_length = params.train_length
        self.pred_length = params.pred_length
        self.nr_sequences = images.shape[0] - self.train_length - self.pred_length
        self.feature_specs = params.feature_specs

        self.dtype = np.dtype(params.dtype)
        self._images = images.astype(self.dtype)

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('ImageDataset index out of range.')

        images = self._images[
            index:index + self.train_length + self.pred_length + 1]

        features = self.to_features(images)
        inputs, labels, pred_labels = utils.split_train_label_pred(
            features, self.train_length, self.pred_length)

        return inputs, labels, pred_labels, images

    def to_features(self, images):
        if images.dtype != self.dtype:
            images = images.astype(self.dtype)
            logger.warning("images dtype is converted to {self.dtype}")

        features = []
        for spec in self.feature_specs:
            if spec["type"] == "pixels":
                _features = pixel_features(images, spec["size"])
            elif spec["type"] == "dct":
                _features = dct_features(images, spec["size"])
            elif spec["type"] == "conv":
                _features = conv_features(images, spec)
            else:
                raise ValueError(spec)
            features.append(_features)

        return np.concatenate(features, axis=1)

    def get_images(self, index):
        images = self._images[
            index:index + self.train_length + self.pred_length + 1]

        _, labels, pred_labels = utils.split_train_label_pred(
            images, self.train_length, self.pred_length)

        return labels, pred_labels

    def __len__(self):
        return self.nr_sequences


# class CircleDataset(NumpyImageDataset):
#     def __init__(self, params, centers=None, sigma=0.5, orig_size=[20, 20]):
# 
#         if centers is None:
#             t = np.arange(0, 200*np.pi, 0.1)
#             x = np.sin(t)
#             y = np.cos(0.25 * t)
#             centers = np.array([y, x]).T
# 
#         images = utils.gauss2d(centers, sigma=sigma, size=orig_size)
#         super(CircleDataset, self).__init__(images, params)


# class TorchImageDataset(Dataset):
#     def __init__(self, asdf):
#         pass
