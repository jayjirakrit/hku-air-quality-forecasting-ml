import torch.nn as nn
import torch


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return nn.functional.relu(out)


class CNN(nn.Module):
    def __init__(self, in_channels, num_residual_units):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)

        layers = []
        for i in range(num_residual_units):
            layers.append(ResidualUnit(in_channels=64, out_channels=64))
            # if i < num_residual_units - 1:
            #     layers.append(nn.Conv2d(64, 64, kernel_size=1))

        self.residual_blocks = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = nn.functional.relu(self.initial_bn(self.initial_conv(x)))
        out = self.residual_blocks(out)
        out = self.adaptive_pool(out)  # (batch_size, 64, 1, 1)
        out = torch.flatten(out, 1)  # (batch_size, 64)
        return out


class AQI_CNNLSTM(nn.Module):
    def __init__(
        self, in_channels, num_residual_units, seq_length, lstm_hidden_size=256, num_lstm_layers=2, pred_len=1
    ):
        super().__init__()
        self.seq_length = seq_length
        self.cnn = CNN(in_channels=in_channels, num_residual_units=num_residual_units)
        self.lstm = nn.LSTM(
            input_size=64,  # Matches the output size of AirRes
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size, pred_len)

    def forward(self, x):
        batch_size, _, channels, height, width = x.shape
        x = x.view(batch_size * self.seq_length, channels, height, width)
        cnn_out = self.cnn(x)  # (batch_size * seq_length, 64)
        lstm_in = cnn_out.view(batch_size, self.seq_length, -1)  # (batch_size, seq_length, 64)
        lstm_out, _ = self.lstm(lstm_in)  # (batch_size, seq_length, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_out)

        return prediction
