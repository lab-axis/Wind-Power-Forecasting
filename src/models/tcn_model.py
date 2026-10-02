"""
TCN (Temporal Convolutional Network) 기반 시계열 예측 모델
- Dilated Causal Convolution을 활용하여 긴 수용 영역(Receptive Field)과 인과성(Causality)을 확보
- 하이브리드 Prophet-TCN 및 단독 TCN 모델 실험용
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """인과성 유지를 위해 Conv 후 시퀀스 끝단의 패딩을 제거하는 레이어"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        forecast_horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, forecast_horizon]
        """
        # [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # 마지막 시점 임베딩 활용
        last_out = y[:, :, -1]
        out = self.fc(last_out)
        return out
