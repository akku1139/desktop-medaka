import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=16, output_size=3):
        super().__init__()
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        # 出力層のバイアスを調整（turn=0, accel=正, interval=ニュートラル）
        with torch.no_grad():
            self.fc.bias[0] = 0.0   # turnはニュートラル
            self.fc.bias[1] = 1.0   # accelを正にバイアス
            self.fc.bias[2] = 0.0   # intervalはニュートラル

    def forward(self, x, h):
        h = self.rnn(x, h)
        out = self.fc(h)
        return out, h
