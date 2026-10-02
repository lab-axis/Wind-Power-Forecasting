"""
CNN-LSTM 기반 풍력발전 시계열 예측 모델
- 박래진·강성우·이재형·정승민 (2022) 「정확도 향상을 위한 CNN-LSTM 기반 풍력발전 예측 시스템」 연계
- 1D-CNN: 다변량 시계열 피처로부터 국소적 시간/변수 패턴 및 특징 추출
- LSTM: 시계열의 중장기 시간적 의존성(temporal dependency) 학습
"""

import torch
import torch.nn as nn


class CNNLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback_steps: int = 24,
        forecast_horizon: int = 1,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon

        # 1D-CNN Feature Extractor (Input: [batch, in_channels, seq_len])
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        # Sequence length after pool
        reduced_len = lookback_steps // 2

        # LSTM Layer (Input: [batch, seq_len, conv_filters])
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Regressor Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
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
        # [batch, seq_len, input_dim] -> [batch, input_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Back to [batch, reduced_seq_len, conv_filters] for LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # 마지막 시점 은닉 상태 활용
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out
