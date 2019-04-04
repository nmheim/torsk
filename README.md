# Torsk

An extended Echo State Network (ESN) for chaotic time series prediction and anomaly
detection.

This is a new implementation of the framework used in my [thesis](https://github.com/nmheim/thesis).
If you are looking for the legacy `torsk` that was used there you 
can find it [here](https://github.com/nmheim/torsk_archived).
In addition to a randomly initialized input matrix this implementation makes it
possible to use convolutions, discrete fourier transforms, and gradients of images
as inputs to the ESN.


## Prediction Examples

For demonstration of the predictive power of the extended ESN
The trajectory of the maximum
governed by the Mackey-Glass time series in the x-dimension and a sine in the
y-dimension. The true evolution of the time series is visible on the left,
the ESN prediction in the middle and the trivial prediction on the right.

### Lissajous Figure

Prediction of a Gaussian blob that moves according to a [Lissajous figure](https://sid.erda.dk/share_redirect/FAtJdDbtah) that is
defined by:

    x = sin(t)
    y = cos(0.3*t)

### Mackey-Glass Lissajous

Prediction of a [chaotically moving Gaussian blob](https://sid.erda.dk/share_redirect/f4ZaRHe9kZ).

### Kuroshio

Prediction of the [Kuroshio](https://sid.erda.dk/share_redirect/ALmNIfYwM5) region
at the coast of Japan.

## Change Backends (WIP)

Switching from Numpy to PyTorch (and soon to
[Bohrium](https://github.com/bh107/bohrium)!) backends can be done by using the
corresponding Numpy/Torch classes. An example usage
which makes it possible to run the prediction with different backends like
this:

    python experiments/chaotic_lissajous/conv_run.py backend torch dtype float32
