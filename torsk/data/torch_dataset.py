import torch
from torch.utils.data import Dataset
from torsk.data.numpy_dataset import split_train_label_pred


class TorchImageDataset(Dataset):
    """Dataset that contains the raw images and does nothing but providing
    convenient access to inputs/labels/pred_labels
    """
    def __init__(self, images, params):
        self.train_length = params.train_length
        self.pred_length = params.pred_length
        self.nr_sequences = images.shape[0] - self.train_length - self.pred_length
        self.max = None
        self.min = None

        self.dtype = getattr(torch, params.dtype)
        _images = torch.tensor(images, dtype=self.dtype)
        self._images = self.scale(_images)
        self.image_shape = images.shape[1:]

    def scale(self, images):
        self.min = torch.min(images)
        self.max = torch.max(images)
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


# class TorchImageDataset(Dataset):
#     def __init__(self, images, params):
#         self._numpy_dataset = NumpyImageDataset(images, params)
#         self.train_length = self._numpy_dataset.train_length
#         self.pred_length = self._numpy_dataset.pred_length
#         self.nr_sequences = self._numpy_dataset.nr_sequences
#         self.feature_specs = self._numpy_dataset.feature_specs
#
#         self.dtype = getattr(torch, params.dtype)
#         self._images = self._numpy_dataset._images
#
#     def __getitem__(self, index):
#         output = self._numpy_dataset[index]
#         output = (torch.tensor(arr[:, None, :], dtype=self.dtype) for arr in output)
#         return output
#
#     def to_features(self, images):
#         features = self._numpy_dataset.to_features(images.numpy())
#         return torch.tensor(features, dtype=self.dtype)
#
#     def get_images(self, index):
#         # TODO: should this return a tensor?
#         #       normally we only call this when plotting...
#         return self._numpy_dataset.get_images(index)
#
#     def to_images(self, features):
#         # TODO: should this return a tensor?
#         #       normally we only call this when plotting...
#         return self._numpy_dataset.to_images(features.numpy())
#
#     def __len__(self):
#         return len(self._numpy_dataset)
