import logging
import numpy as np
import torch
from torsk import utils
from torsk.data.numpy_datasets import NumpyImageDataset

logger = logging.getLogger(__name__)


class TorchImageDataset(torch.data.utils.Dataset):
    def __init_(self, images, params):
        self._numpy_dataset = NumpyImageDataset(images, params)
        self.train_length = self._numpy_dataset.train_length
        self.pred_length = self._numpy_dataset.pred_length
        self.nr_sequences = self._numpy_dataset.nr_sequences
        self.feature_specs = self._numpy_dataset.feature_specs

        self.dtype = getattr(torch, params.dtype)
        self._images = self._numpy_dataset._images

    def __getitem__(self, index):
        inputs, labels, pred_labels, images = self._numpy_dataset[index]
        inputs = torch.tensor(inputs, dtype=self.dtype)
        labels = torch.tensor(labels, dtype=self.dtype)
        pred_labels = torch.tensor(pred_labels, dtype=self.dtype)
        images = torch.tensor(images, dtype=self.dtype)
        return inputs, labels, pred_labels, images

    def to_features(self, images):
        features = self._numpy_dataset.to_features(images.numpy())
        return torch.tensor(features, dtype=self.dtype)

    def get_images(self, index):
        # TODO: should this return a tensor?
        #       normally we only call this when plotting...
        return self._numpy_dataset.get_images(index)

    def __len__(self):
        return len(self._numpy_dataset)
