import numpy as np
from torsk import utils


# DTYPES = ["float32", "float64"]
# BACKENDS = ["numpy", "pytorch"]
DEFAULT_FEATURES = [
    {"type": "pixels", "xsize": [10, 10]},
    {"type": "dct", "ksize": [10, 10]},
    {"type": "conv", "filter": "gauss", "kernel_size": [10, 10]},
    {"type": "conv", "filter": "mean", "kernel_size": [10, 10]},
    {"type": "conv", "filter": "random", "kernel_size": [10, 10]}
]


class NumpyImageDataset:

    def __init__(self, images, params):
        # TODO: use marshmallow instead
        # if dtype not in DTYPES:
        #     raise ValueError(f"`{params.dtype}` is not a valid dtype")
        # if params.backend not in BACKENDS:
        #     raise ValueError(f"`{params.backend}` backend is not available")

        self.backend = params.backend
        self.dtype = np.dtype(params.dtype)
        self.train_length = params.train_length
        self.pred_length = params.pred_length
        self.nr_sequences = images.shape[0] - self.train_length - self.pred_length
        self.feature_specs = params.feature_specs

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

    def __len__(self):
        return self.nr_sequences

    def to_features(self, images):
        if images.dtype != self.dtype:
            raise ValueError(f"images must be dtype `{self.dtype}`")

        seq_len = images.shape[0]
        _features = []

        for spec in self.feature_specs:

            if spec["type"] == "pixels":
                xsize = spec["size"]
                nr_img_features = xsize[0] * xsize[1]
                img_features = utils.resample2d(images, xsize)
                img_features = img_features.reshape([seq_len, nr_img_features])
                _features.append(img_features)

            elif spec["type"] == "dct":
                ksize = spec["size"]
                nr_dct_features = ksize[0] * ksize[1]
                dct_features = utils.dct2_sequence(images, ksize)
                dct_features = dct_features.reshape([seq_len, nr_dct_features])
                _features.append(dct_features)

            elif spec["type"] == "conv":
                kwargs = {k: v for k, v in spec.items() if k != "type"}
                conv_features = utils.conv2d(images, **kwargs)
                kwargs = {k: v for k, v in kwargs.items() if k != "kernel_type"}
                size = utils.conv2d_output_shape(
                    images.shape[1:], **kwargs)
                conv_features = conv_features.reshape([seq_len, size[0] * size[1]])
                _features.append(conv_features)

            else:
                raise ValueError(spec)

        return np.concatenate(_features, axis=1)

    def to_images(self, features):
        types = np.array([spec["type"] for spec in self.feature_specs])
        idx = np.where(types == "pixels")
        if len(idx):
            size = self.feature_specs[0]["size"]
            nr_img_features = size[0] * size[1]
            return features[:, :nr_img_features].reshape([-1, size[0], size[1]])
        else:
            raise NotImplementedError

# class TorchImageDataset(Dataset):
#     def __init__(self, asdf):
#         pass
