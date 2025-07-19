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


class PM_ResidualUnit(nn.Module):
    """Single residual unit with two 3x3 conv layers"""

    def __init__(self, in_channels, out_channels):
        super(PM_ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Skip connection adjustment if input/output channels differ
        self.skip_connection = None
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        # out += identity  # Residual connection
        out = self.relu2(out)

        return out


class PM_CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims=(32, 32), embed_dim=256):
        super().__init__()
        layers = []
        prev_c = in_channels
        for h in hidden_dims:
            layers += [PM_ResidualUnit(prev_c, h)]
            prev_c = h
        self.conv = nn.Sequential(*layers)
        # compute spatial size after pools: 15 → 7 → 3 → 1 (approx)
        self.fc = nn.Linear(prev_c * 15**2, embed_dim)

    def forward(self, x):
        """
        x: (B, C, 15, 15)
        returns: (B, embed_dim)
        """
        z = self.conv(x)  # → (B, H_last, S', S')
        z = z.flatten(1)  # → (B, H_last * S' * S')
        return self.fc(z)  # → (B, embed_dim)


class FSP_CNNLSTM(nn.Module):
    def __init__(self, n_stations, in_channels, cnn_embed=256, lstm_hidden=64, pred_len=24, embed_dim=16):
        super().__init__()
        self.station_emb = nn.Embedding(n_stations, embed_dim)
        self.encoder = PM_CNNEncoder(in_channels, embed_dim=cnn_embed)
        self.lstm = nn.LSTM(cnn_embed + embed_dim, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, pred_len)

    def forward(self, patch_seq, station_idx):
        """
        patch_seq: (B, seq_len, C, 15, 15)
        station_idx: (B,)
        """
        B, T, C, H, W = patch_seq.shape

        # flatten time & batch for encoding
        x = patch_seq.reshape(B * T, C, H, W)
        z = self.encoder(x)  # (B*T, cnn_embed)
        z = z.view(B, T, -1)  # (B, T, cnn_embed)

        # station embedding
        emb = self.station_emb(station_idx)  # (B, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)

        lstm_in = torch.cat([z, emb], dim=-1)  # (B, T, cnn_embed+embed_dim)
        _, (h_n, _) = self.lstm(lstm_in)
        out = self.head(h_n[-1])  # (B, pred_len)
        return out
