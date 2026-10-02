"""
LSTM 및 GRU 기반 시계열 예측 모델 모듈
- 김은지·이택기·김규호 (2021) 「기상 데이터를 이용한 딥러닝 기반 풍력 발전량 예측에 관한 연구」 연계
- 다변량 입력 -> 순환 신경망 -> 다중 horizon 예측
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
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
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        out = self.fc(last_out)
        return torch.relu(out)
