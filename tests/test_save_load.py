import numpy as np
import torch
import torsk
from torsk.models import ESN


def test_save_load(tmpdir):

    params_string = """{
      "input_size": 1,
      "hidden_size": 100,
      "output_size": 1,

      "reservoir_representation": "sparse",
      "spectral_radius" : 2.0,
      "in_weight_init" : 1.00,
      "in_bias_init": 1.00,
      "density": 1e-1,

      "train_length": 800,
      "pred_length": 300,
      "transient_length": 100,
      "train_method": "pinv",
      "tikhonov_beta": 5,

      "sigma": 0.2,
      "xsize": [20,20],
      "ksize": [10,10],
      "domain": "pixels"
    }
    """
    params_json = tmpdir.join("params.json")
    model_pth = tmpdir.join("model.pth")
    with open(params_json, "w") as dst:
        dst.write(params_string)

    params = torsk.Params(params_json)
    model = ESN(params)
    inputs = torch.rand([1, 1, 1])
    state = torch.rand(1, 100)

    _, out1 = model(inputs, state)

    torsk.save_model(model, str(model_pth))

    model = torsk.load_model(str(tmpdir))
    _, out2 = model(inputs, state)

    assert np.all(out1.numpy() == out2.numpy())
