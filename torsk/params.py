import json
import pathlib
from marshmallow import Schema, fields, validate, EXCLUDE


_MODULE_DIR = pathlib.Path(__file__).parent.absolute()


class FeatureSpec(Schema):
    type = fields.String(
        validate=validate.OneOf(["pixels", "dct", "conv"]), required=True)

    # xsize if pixels
    # ksize if dct
    # kernel_shape if conv
    size = fields.List(fields.Int(), required=True)
    kernel_type = fields.String(
        validate=validate.OneOf(["mean", "gauss", "random"]))


class ParamsSchema(Schema):
    input_size = fields.Int(required=True)
    in_weight_init = fields.Float(required=True)
    in_bias_init = fields.Float(required=True)
    feature_specs = fields.List(fields.Nested(FeatureSpec()), required=True)

    hidden_size = fields.Int(required=True)
    reservoir_representation = fields.String(
        validate=validate.OneOf(["sparse", "dense"]), required=True)
    spectral_radius = fields.Float(required=True)
    density = fields.Float(required=True)

    train_length = fields.Int(required=True)
    pred_length = fields.Int(required=True)
    transient_length = fields.Int(required=True)

    train_method = fields.String(
        validate=validate.OneOf(["pinv", "tikhonov"]), required=True)
    tikhonov_beta = fields.Float(missing=None)

    backend = fields.String(
        validate=validate.OneOf(["numpy", "torch"]), required=True)
    dtype = fields.String(
        valdiate=validate.OneOf(["float32", "float64"]), required=True)

    class Meta:
        unkown = EXCLUDE


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path=None, params=None):
        if json_path is not None and params is not None:
            raise ValueError("json_path and params are mutually exclusive args")

        schema = ParamsSchema()

        if json_path is not None:
            with open(json_path) as f:
                self.__dict__ = schema.loads(f.read())

        if params is not None:
            self.__dict__ = schema.load(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            dump = ParamsSchema().dump(self.__dict__)
            json.dump(dump, f, indent=4)

    def update(self, params):
        """Updates parameters based on a dictionary."""
        self.__dict__.update(params)
        self.__dict__ = ParamsSchema().load(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)


def default_params():
    json_path = _MODULE_DIR / "default_params.json"
    return Params(json_path=json_path)
