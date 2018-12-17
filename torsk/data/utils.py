import logging
import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.linalg import lstsq
from scipy.signal import convolve2d
from PIL import Image

logger = logging.getLogger(__name__)


def _mean_kernel(kernel_shape):
    return np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])


def _random_kernel(kernel_shape):
    logger.warning(f"Random kernel created with hard coded uniform dist")
    return np.random.uniform(size=kernel_shape, low=-1, high=1)


def get_kernel(kernel_shape, kernel_type):
    # TODO: gaussian kernel
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_shape)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_shape)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel


def _conv_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    num = in_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return num // stride + 1


def conv2d_output_shape(in_shape, kernel_shape, padding=0, dilation=1, stride=1):
    """Calculate output shape of a convolution of an image of in_shape.
    Formula taken form pytorch conv2d
    """
    height = _conv_out_size(
        in_shape[0], kernel_shape[0], padding, dilation, stride)
    width = _conv_out_size(
        in_shape[1], kernel_shape[1], padding, dilation, stride)
    return (height, width)


def conv2d(sequence, kernel_type, kernel_shape):
    """2D convolution of a sequence of images. Convolution mode is valid, which
    means that only values which do not need to be padded are calculated.

    Params
    ------
    sequence : ndarray
        with shape (time, ydim, xdim)
    kernel_type : str
        one of `gauss`, `mean`, `random`
    kernel_shape : tuple
        shape of created convolution kernel

    Returns
    -------
    ndarray
        convoluted squence of shape (time, height, width). Height and width
        can be calculated with conv2d_output_shape
    """
    kernel = get_kernel(kernel_shape, kernel_type)
    return np.array([convolve2d(img, kernel, mode="valid") for img in sequence])


def resample2d(sequence, size):
    """Resample a squence of 2d-arrays to size using PIL.Image.resize"""
    sequence = [Image.fromarray(img, mode="F") for img in sequence]
    sequence = [np.asarray(img.resize(size)) for img in sequence]
    return np.asarray(sequence)


def sct_basis(nx, nk):
    """Basis for SCT (Slow Cosine Transform) which is a least-squares
    approximation to restricted DCT-III / Inverse DCT-II
    """
    xs = np.arange(nx)
    ks = np.arange(nk)
    basis = 2 * np.cos(np.pi * (xs[:, None] + 0.5) * ks[None, :] / nx)
    return basis


def sct(fx, basis):
    """SCT (Slow Cosine Transform) which is a least-squares approximation to
    restricted DCT-III / Inverse DCT-II
    """
    fk, _, _, _ = lstsq(basis, fx)
    return fk


def isct(fk, basis):
    """Inverse SCT"""
    fx = np.dot(basis, fk)
    return fx


def sct2(Fxx, basis1, basis2):
    """SCT of a two-dimensional array"""
    Fkx = sct(Fxx.T, basis2)
    Fkk = sct(Fkx.T, basis1)
    return Fkk


def isct2(Fkk, basis1, basis2):
    """Inverse SCT of a two-dimensional array"""
    Fkx = isct(Fkk.T, basis2)
    Fxx = isct(Fkx.T, basis1)
    return Fxx


def dct2(Fxx, nk1, nk2):
    """Two dimensional discrete cosine transform"""
    Fkk = dctn(Fxx, norm='ortho')[:nk1, :nk2]
    return Fkk


def idct2(Fkk, nx1, nx2):
    """Two dimensional inverse discrete cosine transform"""
    Fxx = idctn(Fkk, norm='ortho', shape=(nx1, nx2))
    return Fxx


def idct2_sequence(Ftkk, xsize):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftkk : ndarray with shape (time, nk1, nk2)
    size : (ny, nx) determines the resolution of the image

    Returns
    -------
    Ftxx: ndarray with shape (time, ny, nx)
    """
    Ftxx = idctn(Ftkk, norm='ortho', shape=xsize, axes=[1, 2])
    return Ftxx


def dct2_sequence(Ftxx, ksize):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftxx : ndarray with shape (time, ydim, xdim)
    size : (nk1, nk2) determines how many DCT coefficents are kept

    Returns
    -------
    Ftkk: ndarray with shape (time, nk1, nk2)
    """
    Ftkk = dctn(Ftxx, norm='ortho', axes=[1, 2])[:, :ksize[0], :ksize[1]]
    return Ftkk


def normalize(data, vmin=None, vmax=None):
    """Normalizes data to values from 0 to 1.
    If vmin/vmax are given they are assumed to be the maximal
    values of data"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return (data - vmin) / np.abs(vmin - vmax)


def min_max_scale(data, vmin=0., vmax=1.):
    vrange = vmax - vmin
    dmin = data.min()
    drange = data.max() - dmin
    scale = vrange / drange
    shift = vmin - dmin * scale
    data *= scale
    data += shift
    return data


def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels


# def _custom_collate(batch):
#     """Transform batch such that inputs and labels have shape:
#
#         (tot_seq_len, batch_size, nr_features)
#     """
#     def transpose(tensor):
#         return torch.transpose(torch.stack(tensor), 0, 1)
#     batch = [list(b) for b in zip(*batch)]
#     batch = [transpose(b) for b in batch]
#     return batch
#
#
# class SeqDataLoader(DataLoader):
#     """Custom Dataloader that defines a fixed custom collate function, so that
#     the loader returns batches of shape (seq_len, batch, nr_features).
#     """
#     def __init__(self, dataset, **kwargs):
#         if 'collate_fn' in kwargs:
#             raise ValueError(
#                 'SeqDataLoader does not accept a custom collate_fn '
#                 'because it already implements one.')
#         kwargs['collate_fn'] = _custom_collate
#         super(SeqDataLoader, self).__init__(dataset, **kwargs)
