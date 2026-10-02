"""
DLinear (Decomposition Linear) 시계열 예측 모델 모듈
- Zeng et al. (AAAI 2023) "Are Transformers Effective for Time Series?"
- 최근 SCI Q1급 시계열 및 에너지 예측 학계에서 가장 강력하고 필수적인 벤치마크 선형 분해 모델
- 이동평균(Moving Average) 기반 Trend-Seasonal 분해 + Dual Linear Mapping
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """이동평균 기반 추세(Trend) 추출 레이어"""

    def __init__(self, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, num_features] -> [batch, num_features, seq_len]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.permute(0, 2, 1)
        trend = self.avg(x_pad)
        trend = trend.permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    """시계열 분해: x = Seasonal + Trend"""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    """
    DLinear: Decomposition Linear Model for Time Series
    """

    def __init__(
        self,
        lookback_steps: int = 24,
        forecast_horizon: int = 1,
        input_dim: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

        # 시계열 분해
        self.decompsition = SeriesDecomp(kernel_size=kernel_size)

        # 계절성 및 추세 각각에 대한 선형 변환
        self.linear_seasonal = nn.Linear(lookback_steps * input_dim, forecast_horizon)
        self.linear_trend = nn.Linear(lookback_steps * input_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback_steps, input_dim]
        Returns:
            [batch, forecast_horizon]
        """
        batch_size = x.size(0)

        # 1. 분해
        seasonal_init, trend_init = self.decompsition(x)

        # 2. 평탄화 [batch, lookback_steps * input_dim]
        seasonal_flat = seasonal_init.reshape(batch_size, -1)
        trend_flat = trend_init.reshape(batch_size, -1)

        # 3. 선형 매핑
        seasonal_output = self.linear_seasonal(seasonal_flat)
        trend_output = self.linear_trend(trend_flat)

        # 4. 결합
        x_out = seasonal_output + trend_output
        return x_out
