import numpy as np
from scipy.signal import convolve2d
from torsk.imed import imed_metric, metric_matrix


def gauss_kernel(kernel_shape, sigma=None):
    if sigma is None:
        sigma = min(kernel_shape) / 5.
    ysize, xsize = kernel_shape
    yy = np.linspace(-int(ysize / 2.), int(ysize / 2.), ysize)
    xx = np.linspace(-int(xsize / 2.), int(xsize / 2.), xsize)

    P = xx[:, None]**2 + yy[None, :]**2
    gaussian = np.exp(-P / (2 * sigma**2))
    gaussian = 1. / (2 * np.pi * sigma**2) * gaussian
    return gaussian


def test_imed():
    np.random.seed(0)

    img1 = np.random.normal(size=[5,5])
    img2 = np.random.normal(size=[5,5])
    diff = img1 - img2

    kernel = gauss_kernel([5,5], sigma=0.5)
    conv_diff = convolve2d(diff, kernel, mode="same")

    G = metric_matrix(diff.shape, sigma=0.5)
    G_diff = G.dot(diff.reshape(-1)).reshape(diff.shape)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(kernel)
    # ax[1].imshow(G)

    # fig2, ax2 = plt.subplots(1,2)
    # ax2[0].imshow(conv_diff)
    # ax2[1].imshow(G_diff)
    # plt.show()

    assert np.allclose(conv_diff, G_diff)

    imed = imed_metric(img1[None,:,:], img2[None,:,:], G=G)
    conv_diff = (diff * conv_diff).sum()

    assert np.allclose(imed, conv_diff)
