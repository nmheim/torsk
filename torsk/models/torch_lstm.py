import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = self.out(x)
        return x

    def predict(self, inputs, steps):

        lstm_out, state = self.lstm(inputs)
        x = self.out(lstm_out)
        x = x[:, -1].unsqueeze(dim=1)

        pred = []
        for ii in range(steps):
            x, state = self.lstm(x, state)
            p = self.out(x)
            x = p
            pred.append(p)
        pred = torch.cat(pred, dim=1)
        return pred
