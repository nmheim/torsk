# TORSK (WIP)!

An Echo State Network (ESN) for chaotic time series prediction and anomaly
detection.

This is a new implementation of the framework used in my [thesis](https://github.com/nmheim/thesis).
If you are looking for the legacy `torsk` that was used there you 
can find it [here](https://github.com/nmheim/torsk_archived)

## Prediction Examples

### Mackey-Glass Lissajous

Prediction of a [chaotically moving Gaussian blob](https://sid.erda.dk/share_redirect/EXaVvPLNAq).
The trajectory of the maximum
governed by the Mackey-Glass time series in the x-dimension and a sine in the
y-dimension. The true evolution of the time series is visible on the left,
the ESN prediction in the middle and the trivial prediction on the right.

### Kuroshio

Prediction of the [Kuroshio](https://sid.erda.dk/share_redirect/ALmNIfYwM5) region
at the coast of Japan.

## Backend

Switching from Numpy to PyTorch (and soon to
[Bohrium](https://github.com/bh107/bohrium)!) backends can be done by using the
corresponding Numpy/Torch classes. The circle script shows an example usage
which makes it possible to run the prediction with different backends like
this:

    python experiments/circle/run.py backend torch dtype float64

The same works for switching between `float32` and `float64`
