import torch
from torch import nn
import convlstm


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


class ConvLSTM(nn.Module):

    def __init__(self, input_width, input_height, hidden_channels,
            kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        input_channels = 1
        self.lstm = convlstm.ConvLSTM(
            input_channels, hidden_channels, kernel_size, num_layers=1,
            batch_first=True)

        input_size = input_width * input_height
        hidden_size = hidden_channels * input_size
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, inputs):
        B, T, H, W = inputs.size()
        x = torch.reshape(inputs, (B,T,1,H,W))
        x = self.lstm(x)[0][0]
        x = torch.reshape(x, (B,T,-1))
        x = self.out(x)
        x = torch.reshape(x, (B,T,H,W))
        return x

    def predict(self, inputs, steps):
        B, T, H, W = inputs.size()
        x = torch.reshape(inputs, (B,T,1,H,W))
        output, state = self.lstm(x)
        output = output[0]

        output = torch.reshape(output, (B,T,-1))
        output = self.out(output)
        output = torch.reshape(output, (B,T,H,W))

        x = output[:, -1].unsqueeze(dim=1)
        B, T, H, W = x.size()
        pred = []
        for ii in range(steps):
            x = torch.reshape(x, (B,T,1,H,W))
            output, state = self.lstm(x, state)
            output = output[0]

            output = torch.reshape(output, (B,T,-1))
            output = self.out(output)
            output = torch.reshape(output, (B,T,H,W))

            x = output
            pred.append(output)
        pred = torch.cat(pred, dim=1)
        return pred


