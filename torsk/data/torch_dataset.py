import logging
import torch
from torch.utils.data import Dataset
from torsk.data.numpy_dataset import NumpyImageDataset

logger = logging.getLogger(__name__)


class TorchImageDataset(Dataset):
    def __init__(self, images, params):
        self._numpy_dataset = NumpyImageDataset(images, params)
        self.train_length = self._numpy_dataset.train_length
        self.pred_length = self._numpy_dataset.pred_length
        self.nr_sequences = self._numpy_dataset.nr_sequences
        self.feature_specs = self._numpy_dataset.feature_specs

        self.dtype = getattr(torch, params.dtype)
        self._images = self._numpy_dataset._images

    def __getitem__(self, index):
        output = self._numpy_dataset[index]
        output = (torch.tensor(arr[:, None, :], dtype=self.dtype) for arr in output)
        return output

    def to_features(self, images):
        features = self._numpy_dataset.to_features(images.numpy())
        return torch.tensor(features, dtype=self.dtype)

    def get_images(self, index):
        # TODO: should this return a tensor?
        #       normally we only call this when plotting...
        return self._numpy_dataset.get_images(index)

    def to_images(self, features):
        # TODO: should this return a tensor?
        #       normally we only call this when plotting...
        return self._numpy_dataset.to_images(features.numpy())

    def __len__(self):
        return len(self._numpy_dataset)
