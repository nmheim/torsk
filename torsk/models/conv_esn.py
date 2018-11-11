import torch
import torch.nn as nn

from torsk.models.esn import ESN


class ConvESN(nn.Module):

    def __init__(self, params):
        self.params = params
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=params.nr_filters,
            kernel_size=params.kernel_size)
        self.esn = ESN(params)

    def forward(self, inputs, nr_predictions=0):
        if inputs.size(1) != 1:
            raise ValueError("Supports only batch size of one -.-")

        raise NotImplementedError("Add _forward fuctions like in ESN class")

        act = self.conv(inputs)
        act = output.reshape([inputs.size(0), 1, -1])
        state = torch.zeros(1, params.hidden_size)
        act, _ = self.esn(act, state, nr_predictions)
        act = act.reshape([-1, 1, inputs.size(2), inputs.size(3)])
        return act
