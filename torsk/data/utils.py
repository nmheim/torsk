import numpy as np
import skimage.transform as skt
from scipy.fftpack import dct, idct, fft, ifft, fftshift, ifftshift;

def resample2d(image, size):
    res = skt.resize(image, size, mode="reflect", anti_aliasing=True)
    return res.astype(image.dtype)


def resample2d_sequence(sequence, size):
    """Resample a squence of 2d-arrays to size using PIL.Image.resize"""
    dtype = sequence.dtype
    sequence = [skt.resize(img, size, mode="reflect", anti_aliasing=True)
                for img in sequence]
    return np.asarray(sequence, dtype=dtype)

def upscale(ft,nX):
    ct=dct(ft,norm='ortho');
    return idct(ct,n=nX,norm='ortho');

def downscale(Ft,nx):
    ct=dct(Ft,norm='ortho')[:nx];
    return idct(ct,norm='ortho');

def fft_derivative_1d(fx,order=1):
    N  = len(fx)
    ks = np.linspace(-N/2,N/2,N,endpoint=False)**order;    
    Fk = ifft(fx);
    fftpack.fftshift(Fk)
    DFk = fftpack.ifftshift((2j*pi/N)*ks*Fk)
    DFx = fftpack.fft((2j*np.pi/N)*ks*Fk)
    
    return DFx;


def normalize(data, vmin=None, vmax=None):
    """Normalizes data to values from 0 to 1.
    If vmin/vmax are given they are assumed to be the maximal
    values of data"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    if vmin==vmax:
        return np.zeros_like(data)
    else:
        return (data - vmin) / (vmax-vmin)


def min_max_scale(data, vmin=0., vmax=1.):
    vrange = vmax - vmin
    dmin   = data.min()
    drange = data.max() - dmin

    if drange == 0:
        data += dmin
    else:
        scale = vrange / drange
        shift = vmin - dmin * scale
        data *= scale
        data += shift

    return data


def gauss2d_sequence(centers=None, sigma=0.5, size=[20, 20], borders=[[-2, 2], [-2, 2]]):
    """Creates a moving gaussian blob on grid with `size`"""
    if centers is None:
        t = np.arange(0, 200 * np.pi, 0.1)
        x = np.sin(t)
        y = np.cos(0.25 * t)
        centers = np.array([y, x]).T

    yc, xc = centers[:, 0], centers[:, 1]
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])

    xx = xx[None, :, None] - xc[:, None, None]
    yy = yy[None, None, :] - yc[:, None, None]

    gauss = (xx**2 + yy**2) / (2 * sigma**2)
    return np.exp(-gauss)


def mackey_sequence(b=None, N=3000):
    """Create the Mackey-Glass series"""
    c = 0.2
    tau = 17
    n = 10

    yinit = np.array([0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076,
        1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
        1.0739, 1.0759])

    if b is None:
        b = np.zeros(N) + 0.1

    y = np.zeros(N)
    y[:yinit.shape[0]] = yinit

    for i in range(tau, N - 1):
        yi = y[i] - b[i] * y[i] + c * y[i - tau] / (1 + y[i - tau]**n)
        y[i + 1] = yi
    return y


def mackey_anomaly_sequence(N=3000, anomaly_start=2000, anomaly_step=200):
    b = np.zeros(N) + 0.1
    db = 0.05
    anomaly = np.zeros(N)
    for i in range(anomaly_start, N, anomaly_step):
        b[i:i+50] += db
        anomaly[i:i+50] = 1
    return mackey_sequence(b, N=N), anomaly


def sine_sequence(periods=30, N=20):
    """Simple sine sequence"""
    dx = 2 * np.pi / (N + 1)
    x = np.linspace(0, 2 * np.pi - dx, N)
    y = np.sin(x)
    y = np.tile(y, periods)
    return y
