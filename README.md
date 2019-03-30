# TORSK (WIP)!

An Echo State Network (ESN) for chaotic time series prediction and anomaly
detection.

This is a new implementation of the framework used in my [thesis](https://github.com/nmheim/thesis).
If you are looking for the legacy `torsk` that was used there you 
can find it [here](https://github.com/nmheim/torsk_archived)

## Prediction Examples

### Mackey-Glass Lissajous

Prediction of a [chaotically moving Gaussian blob](https://sid.erda.dk/share_redirect/eosH8APA8K). The trajectory of the maximum
governed by the Mackey-Glass time series in the x-dimension and a sine in the
y-dimension. The true evolution of the time series is visible on the left,
the ESN prediction in the middle and the trivial prediction on the right.

### Kuroshio

![here](https://sid.erda.dk/share_redirect/hEpNhYk1gs)

## Backend

Switching from Numpy to PyTorch (and soon to
[Bohrium](https://github.com/bh107/bohrium)!) backends can be done by using the
corresponding Numpy/Torch classes. The circle script shows an example usage
which makes it possible to run the prediction with different backends like
this:

    python experiments/circle/run.py backend torch dtype float64

The same works for switching between `float32` and `float64`


## Params

The commands override the default parameters. Valid fields for the
torch.Params class are defined with `marshmallow.Schema`s. Schemas are used to
load a dictionary and make sure that only fields that are defined by the Schema
are used as parameters for the network.

For example the `torsk.params.FeatureSpec` schema looks like this:

    class FeatureSpec(Schema):
        type = fields.String(
            validate=validate.OneOf(["pixels", "dct", "conv"]), required=True)
        size = fields.List(fields.Int(), required=True)
        kernel_type = fields.String(
            validate=validate.OneOf(["mean", "gauss", "random"]))

This means that a feature spec has two required keys: `type` and `size`.
The `type` can be one of `pixels`, ... The `size` is used as the resample size
(former `xsize`) if the `type` is `pixels`, as `ksize` if `dct`, and as
`kernel_shape` if `conv`.  The optional `kernel_type` is only used if the type
is `conv` and specifies what kind of kernel the convolution is using.


## Input features

(not throughly tested that all of this works correctly all the time, but the
infrastructure is there)

The stack of input features is created with the `torsk.data.NumpyImageDataset`
which reads what it needs to do from given `params.feature_specs`.  The
`params.feature_specs` is a list dicts that are created based on the
`FeatureSpec` schema.  The idea is that you can supply as many such specs as
you want and the dataset will glue them all together.

Example params:

    params = {
        ...
        feature_specs: [
            {"type": "pixels", "size":[30, 30]},
            {"type": "conv", "size":[5, 5], "kernel_type": "gauss"},
            {"type": "dct", "size":[10, 10]},
        ]
    }

The standard deviation of the gaussian kernels is fixed to `min(kernel_shape)/4`
so the Gaussian is approximately zero at the edges of the kernel window.
The dataset does all the preprocessing that is defined via the feature specs
and then flattens the resulting stack of images to a single input vector.

Example based on the feature specs above:

    raw_image[100x100] -> [resampled_image[30x30], conv_image[96x96], dct_coeffs[30x30]]
                       -> [feature_vec[11016]]


# ESN

Currently the `params.input_size` has to be set manually to the value size
resulting from the feature specs and the ESN will always contain an input layer
and a hidden layer. We could think about leaving out the input layer completely,
but then we still have to think about a scaling that we apply to the individual
input feature specs.


# Anomaly detection

To be filled out...
