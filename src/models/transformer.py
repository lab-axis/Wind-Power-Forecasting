"""
시계열 트랜스포머(Time Series Transformer) 예측 모델 모듈
- Vaswani et al. (2017) Transformer 아키텍처의 시계열 적응 모델
- Informer (AAAI 2021 Best Paper), Autoformer (NeurIPS 2021) 등 수많은 SCI Q1급 시계열 논문들의 기준 베이스라인
- Positional Encoding + Multi-Head Self-Attention + Feed-Forward Network
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformerForecaster(nn.Module):
    """
    상명풍력 시계열 예측용 Vanilla Transformer 모델
    """

    def __init__(
        self,
        input_dim: int = 1,
        lookback_steps: int = 24,
        forecast_horizon: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon

        # 입력 투영 레이어 (input_dim -> d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=lookback_steps + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Regressor Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, forecast_horizon]
        """
        # 1. 입력 투영 및 위치 인코딩
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        # 2. 트랜스포머 인코더 멀티헤드 어텐션
        out = self.transformer_encoder(h)

        # 3. 마지막 시점 표현(Representation) 추출 및 발전량 회귀
        last_step = out[:, -1, :]
        pred = self.fc(last_step)
        return pred
