import torch
from torsk.models.torch_lstm import ConvLSTM, LSTM


def test_lstm():
    batch = 100
    input_size = 10
    hidden_size = 20
    series_length = 30

    x = torch.rand(batch, series_length, input_size)
    m = LSTM(input_size, hidden_size)
    h, _ = m.lstm.forward(x)
    y = m.forward(x)

    assert h.size() == torch.Size([batch, series_length, hidden_size])
    assert y.size() == torch.Size([batch, series_length, input_size])

def test_convlstm():
    import convlstm
    hidden_size = 20
    kernel_size = (15,15)
    B, T, C, H, W = 32, 10, 1, 30, 30

    m = ConvLSTM(W, H, hidden_size, kernel_size)
    x = torch.rand((B, T, C, H, W))
    y = m(x)
    assert y.size() == torch.Size([B, H, W])

if __name__ == "__main__":
    test_lstm()
    test_convlstm()
