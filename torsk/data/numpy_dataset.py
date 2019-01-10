import logging
import numpy as np

from torsk.data.utils import resample2d_sequence, normalize
from torsk.data.conv import conv2d_sequence, conv2d_output_shape
from torsk.data.dct import dct2_sequence

logger = logging.getLogger(__name__)


def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels


class NumpyImageDataset:
    """Dataset that contains the raw images and does nothing but providing
    convenient access to inputs/labels/pred_labels
    """
    def __init__(self, images, params):
        self.train_length = params.train_length
        self.pred_length = params.pred_length
        self.nr_sequences = images.shape[0] - self.train_length - self.pred_length
        self.max = None
        self.min = None

        self.dtype = np.dtype(params.dtype)
        _images = self.scale(images)
        self._images = _images.astype(self.dtype)
        self.image_shape = images.shape[1:]

    def scale(self, images):
        self.min = images.min()
        self.max = images.max()
        normalized = (images - self.min) / (self.max - self.min)
        scaled = normalized * 2 - 1
        return scaled

    def unscale(self, images):
        if self.max is None or self.min is None:
            raise ValueError("Min/max not set. Call 'scale' first.")
        normalized = (images + 1) * 0.5
        orig = normalized * (self.max - self.min) + self.min
        return orig

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('ImageDataset index out of range.')

        images = self._images[
            index:index + self.train_length + self.pred_length + 1]

        inputs, labels, pred_labels = split_train_label_pred(
            images, self.train_length, self.pred_length)

        return inputs, labels, pred_labels

    def __len__(self):
        return self.nr_sequences


# def pixel_feature(image, xsize):
#     nr_pxl_features = xsize[0] * xsize[1]
# 
# 
# def pixel_features(images, xsize):
#     """Resample a sequence of images to xsize and reshape them to a sequence of
#     feature vectors.
#     """
#     seq_len = images.shape[0]
#     nr_pxl_features = xsize[0] * xsize[1]
#     pxl_features = resample2d_sequence(images, xsize)
#     return pxl_features.reshape([seq_len, nr_pxl_features])
# 
# 
# def dct_features(images, ksize):
#     """Apply 2d DCT to images, keep only window of ksize coefficients and
#     create feature vector sequence.
#     """
#     seq_len = images.shape[0]
#     nr_dct_features = ksize[0] * ksize[1]
#     dct_features = dct2_sequence(images, ksize)
#     return dct_features.reshape([seq_len, nr_dct_features])
# 
# 
# def conv_features(images, spec):
#     """Apply 2d convolution specified by spec and reshape to a sequence of
#     feature vecotrs.
#     """
#     seq_len = images.shape[0]
#     kwargs = {k: v for k, v in spec.items() if k != "type"}
#     conv_features = conv2d_sequence(images, **kwargs)
#     kwargs = {k: v for k, v in kwargs.items() if k != "kernel_type"}
#     size = conv2d_output_shape(images.shape[1:], **kwargs)
#     return conv_features.reshape([seq_len, size[0] * size[1]])
# 
# 
# class NumpyImageDataset:
# 
#     def __init__(self, images, params):
#         self.train_length = params.train_length
#         self.pred_length = params.pred_length
#         self.nr_sequences = images.shape[0] - self.train_length - self.pred_length
#         self.input_map_specs = params.input_map_specs
# 
#         self.dtype = np.dtype(params.dtype)
#         self._images = images.astype(self.dtype)
# 
#     def __getitem__(self, index):
#         if (index < 0) or (index >= self.nr_sequences):
#             raise IndexError('ImageDataset index out of range.')
# 
#         images = self._images[
#             index:index + self.train_length + self.pred_length + 1]
# 
#         features = self.to_features(images)
#         inputs, labels, pred_labels = split_train_label_pred(
#             features, self.train_length, self.pred_length)
# 
#         return inputs, labels, pred_labels, images
# 
#     def to_features(self, images):
#         if images.dtype != self.dtype:
#             images = images.astype(self.dtype)
#             logger.warning("images dtype is converted to {self.dtype}")
# 
#         features = []
#         for spec in self.input_map_specs:
#             if spec["type"] == "pixels":
#                 _features = pixel_features(images, spec["size"])
#             elif spec["type"] == "dct":
#                 _features = dct_features(images, spec["size"])
#             elif spec["type"] == "conv":
#                 _features = conv_features(images, spec)
#             else:
#                 raise ValueError(spec)
#             features.append(_features)
# 
#         return np.concatenate(features, axis=1)
# 
#     def get_raw_images(self, index):
#         images = self._images[
#             index:index + self.train_length + self.pred_length + 1]
#         inputs, labels, pred_labels = split_train_label_pred(
#             images, self.train_length, self.pred_length)
#         return inputs, labels, pred_labels
# 
#     def to_images(self, features):
#         types = np.array([spec["type"] for spec in self.input_map_specs])
#         if "pixels" in types:
#             idx = np.where(types == "pixels")[0][0]
#             size = self.input_map_specs[idx]["size"]
#             nr_pixel_features = size[0] * size[1]
#             images = features[:, :nr_pixel_features]
#             images = images.reshape([-1, size[0], size[1]])
#         elif "dct" in types:
#             raise NotImplementedError
#         else:
#             raise ValueError(f"Cannot reconstruct images from feature types {types}")
#         return images
# 
#     def __len__(self):
#         return self.nr_sequences
# 
# 
# class NumpyScalarDataset:
#     def __init__(self, sequence, params):
#         self.train_length = params.train_length
#         self.pred_length = params.pred_length
#         self.nr_sequences = sequence.shape[0] - self.train_length - self.pred_length
# 
#         self.seq = normalize(sequence[:, None])
# 
#     def __getitem__(self, index):
#         if (index < 0) or (index >= self.nr_sequences):
#             raise IndexError('MackeyDataset index out of range.')
#         sub_seq = self.seq[index:index + self.train_length + self.pred_length + 1]
#         inputs, labels, pred_labels = split_train_label_pred(
#             sub_seq, self.train_length, self.pred_length)
#         return inputs, labels, pred_labels
# 
#     def __len__(self):
#         return self.nr_sequences
