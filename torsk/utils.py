import logging
import pathlib
import numpy as np
import netCDF4 as nc
from scipy.fftpack import dctn, idctn
from scipy.linalg import lstsq
from scipy.signal import convolve2d
# from PIL import Image
import skimage.transform as skt

logger = logging.getLogger(__name__)


def gauss2d(centers, sigma, size, borders=[[-2, 2], [-2, 2]]):
    yc, xc = centers[:, 0], centers[:, 1]
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])

    xx = xx[None, :, None] - xc[:, None, None]
    yy = yy[None, None, :] - yc[:, None, None]

    gauss = (xx**2 + yy**2) / (2 * sigma**2)
    return np.exp(-gauss)


def _mean_kernel(kernel_shape):
    return np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])


def _random_kernel(kernel_shape):
    logger.warning(f"Random kernel created with hard coded uniform dist")
    return np.random.uniform(size=kernel_shape, low=-1, high=1)


def _gauss_kernel(kernel_shape):
    ysize, xsize = kernel_shape
    yy = np.linspace(-ysize/2., ysize/2., ysize)
    xx = np.linspace(-xsize/2., xsize/2., xsize)
    sigma = min(kernel_shape) / 4.
    yy, xx = np.meshgrid(yy, xx)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def get_kernel(kernel_shape, kernel_type):
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_shape)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_shape)
    elif kernel_type == "gauss":
        kernel = _gauss_kernel(kernel_shape)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel


def _conv_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    num = in_size + 2 * padding - dilation * (kernel_size - 1) - 1
    size = num // stride + 1
    return size


def conv2d_output_shape(in_size, size, padding=0, dilation=1, stride=1):
    """Calculate output shape of a convolution of an image of in_shape.
    Formula taken form pytorch conv2d
    """
    height = _conv_out_size(in_size[0], size[0], padding, dilation, stride)
    width = _conv_out_size(in_size[1], size[1], padding, dilation, stride)
    return (height, width)


def conv2d(sequence, kernel_type, size):
    """2D convolution of a sequence of images. Convolution mode is valid, which
    means that only values which do not need to be padded are calculated.

    Params
    ------
    sequence : ndarray
        with shape (time, ydim, xdim)
    kernel_type : str
        one of `gauss`, `mean`, `random`
    size : tuple
        shape of created convolution kernel

    Returns
    -------
    ndarray
        convoluted squence of shape (time, height, width). Height and width
        can be calculated with conv2d_output_shape
    """
    kernel = get_kernel(size, kernel_type)
    conv = [convolve2d(img, kernel, mode="valid") for img in sequence]
    return np.asarray(conv, dtype=sequence.dtype)


def resample2d(sequence, size):
    """Resample a squence of 2d-arrays to size using PIL.Image.resize"""
    dtype = sequence.dtype
    sequence = [skt.resize(img, size, mode="reflect", anti_aliasing=True)
                for img in sequence]
    return np.asarray(sequence, dtype=dtype)


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

def _fix_prefix(prefix):
    if prefix is not None:
        prefix = prefix.strip("-") + "-"
    else:
        prefix = ""
    return prefix


# def create_training_states(model, inputs):
#     zero_state = torch.zeros(1, model.params.hidden_size)
#     _, states = model(inputs, zero_state, states_only=True)
#     return states


def save_model(modeldir, model, prefix=None):
    if not isinstance(modeldir, pathlib.Path):
        modeldir = pathlib.Path(modeldir)
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
    prefix = _fix_prefix(prefix)

    model_pth = modeldir / f"{prefix}model.pth"
    params_json = modeldir / f"{prefix}params.json"
    state_dict = model.state_dict()

    # convert sparse tensor
    key = "esn_cell.weight_hh"
    if isinstance(state_dict[key], torch.sparse.FloatTensor):
        # TODO: can be removed when save/load is implemented for sparse tensors
        # discussion: https://github.com/pytorch/pytorch/issues/9674
        weight = state_dict.pop(key)
        state_dict[key + "_indices"] = weight.coalesce().indices()
        state_dict[key + "_values"] = weight.coalesce().values()

    model.params.save(params_json.as_posix())
    torch.save(state_dict, model_pth.as_posix())


def load_model(modeldir, prefix=None):
    # TODO: fix circular import
    from torsk.models import ESN
    if isinstance(modeldir, str):
        modeldir = pathlib.Path(modeldir)
    prefix = _fix_prefix(prefix)

    params = torsk.Params(modeldir / f"{prefix}params.json")
    model = ESN(params)
    state_dict = torch.load(modeldir / f"{prefix}model.pth")

    # restore sparse tensor
    key = "esn_cell.weight_hh"
    key_idx = key + "_indices"
    key_val = key + "_values"
    if key_idx in state_dict:
        # TODO: can be removed when save/load is implemented for sparse tensors
        # discussion: https://github.com/pytorch/pytorch/issues/9674
        weight_idx = state_dict.pop(key_idx)
        weight_val = state_dict.pop(key_val)
        hidden_size = params.hidden_size
        weight_hh = torch.sparse.FloatTensor(
            weight_idx, weight_val, [hidden_size, hidden_size])
        state_dict[key] = weight_hh

    model.load_state_dict(state_dict)

    return model


def dump_training(fname, inputs, labels, states, pred_labels, attrs=None):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.numpy()

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("train_length", inputs.shape[0])
        dst.createDimension("pred_length", pred_labels.shape[0])
        dst.createDimension("inputs_size", inputs.shape[1])
        dst.createDimension("outputs_size", labels.shape[1])
        dst.createDimension("hidden_size", states.shape[1])

        dst.createVariable("inputs", float, ["train_length", "inputs_size"])
        dst.createVariable("labels", float, ["train_length", "outputs_size"])
        dst.createVariable("states", float, ["train_length", "hidden_size"])
        dst.createVariable("pred_labels", float, ["pred_length", "outputs_size"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["inputs"][:] = inputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["pred_labels"][:] = pred_labels


def dump_prediction(fname, outputs, labels, states, attrs=None):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(states, torch.Tensor):
        states = states.numpy()

    if not isinstance(fname, pathlib.Path):
        fname = pathlib.Path(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    error = (outputs - labels)**2
    rmse = np.mean(error)**.5

    with nc.Dataset(fname, "w") as dst:

        dst.createDimension("pred_length", outputs.shape[0])
        dst.createDimension("output_size", outputs.shape[1])
        dst.createDimension("hidden_size", states.shape[1])
        dst.createDimension("scalar", 1)

        dst.createVariable("outputs", float, ["pred_length", "output_size"])
        dst.createVariable("labels", float, ["pred_length", "output_size"])
        dst.createVariable("states", float, ["pred_length", "hidden_size"])
        dst.createVariable("rmse", float, ["scalar"])

        if attrs is not None:
            dst.setncatts(attrs)

        dst["outputs"][:] = outputs
        dst["labels"][:] = labels
        dst["states"][:] = states
        dst["rmse"][:] = rmse


def create_path(root, param_dict, prefix=None, postfix=None):
    if not isinstance(root, pathlib.Path):
        root = pathlib.Path(root)
    folder = _fix_prefix(prefix)
    for key, val in param_dict.items():
        folder += f"{key}:{val}-"
    folder = folder[:-1]
    if postfix is not None:
        folder += f"-{postfix}"
    return root / folder


def parse_path(path):
    param_dict = {}
    for string in path.name.split("-"):
        if ":" in string:
            key, val = string.split(":")
            try:
                val = eval(val)
            except Exception:
                pass
            param_dict[key] = val
    return param_dict
