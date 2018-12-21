import numpy as np
import torsk
from torsk.models.numpy_esn import NumpyESN


def test_numpy_save_load(tmpdir):

    params_string = """{
      "input_size": 1,
      "hidden_size": 100,

      "reservoir_representation": "dense",
      "spectral_radius" : 2.0,
      "in_weight_init" : 1.00,
      "in_bias_init" : 1.00,
      "density": 1e-1,

      "train_length": 800,
      "pred_length": 300,
      "transient_length": 100,
      "train_method": "pinv",

      "dtype": "float64",
      "backend": "numpy",
      "feature_specs": [{"type":"pixels", "size":[1, 1]}]
    }
    """
    params_json = tmpdir.join("params.json")
    with open(params_json, "w") as dst:
        dst.write(params_string)

    params = torsk.Params(params_json)
    model = NumpyESN(params)
    inputs = np.random.uniform(size=[1, 1])
    state = np.random.uniform(size=[100])

    _, out1 = model.forward(inputs, state)

    torsk.save_model(tmpdir, model)

    model = torsk.load_model(str(tmpdir))
    _, out2 = model.forward(inputs, state)

    assert np.all(out1 == out2)


def test_torch_save_load(tmpdir):
    import torch
    from torsk.models.torch_esn import TorchESN

    params_string = """{
      "input_size": 1,
      "hidden_size": 100,

      "reservoir_representation": "dense",
      "spectral_radius" : 2.0,
      "in_weight_init" : 1.00,
      "in_bias_init" : 1.00,
      "density": 1e-1,

      "train_length": 800,
      "pred_length": 300,
      "transient_length": 100,
      "train_method": "pinv",

      "dtype": "float32",
      "backend": "torch",
      "feature_specs": [{"type":"pixels", "size":[1, 1]}]
    }
    """
    params_json = tmpdir.join("params.json")
    with open(params_json, "w") as dst:
        dst.write(params_string)

    params = torsk.Params(params_json)
    model = TorchESN(params)
    inputs = torch.rand(1, 1, 1)
    state = torch.rand(1, 100)

    _, out1 = model(inputs, state)

    torsk.save_model(tmpdir, model)

    model = torsk.load_model(str(tmpdir))
    _, out2 = model(inputs, state)

    assert np.all(out1.numpy() == out2.numpy())
