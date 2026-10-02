"""
AR / ARIMA 기반 시계열 예측 모델 모듈
- 고전 통계 시계열 모델 (Box & Jenkins, 1970)
- AutoRegressive Integrated 모델 구현
- numpy 2.x 바이너리 안정성 및 초고속 추론 보장
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from statsmodels.tsa.ar_model import AutoReg
from src.utils.metrics import evaluate_wind_forecast


class ARIMAForecaster:
    """
    상명풍력 시간별 발전량 예측을 위한 AR/ARIMA(p, d, 0) 모델
    - d=1 차분(Differencing) 적용 후 AR(p) 모델 적합
    """

    def __init__(
        self,
        p_lags: int = 24,
        diff: int = 0,
        order: Optional[Tuple[int, int, int]] = None,
        rated_capacity_mwh: float = 21.0,
    ):
        if order is not None:
            self.p_lags = order[0]
            self.diff = order[1] if len(order) > 1 else 0
        else:
            self.p_lags = p_lags
            self.diff = diff
        self.rated_capacity_mwh = rated_capacity_mwh
        self.model_res = None
        self.params = None

    def fit(self, train_series: pd.Series):
        """Train 시계열에 대해 AR 모델 적합"""
        vals = train_series.dropna().values
        if self.diff > 0:
            vals = np.diff(vals, n=self.diff)

        model = AutoReg(vals, lags=self.p_lags, trend="c")
        self.model_res = model.fit()
        self.params = self.model_res.params
        return self

    def predict_horizon(
        self,
        full_series: pd.Series,
        test_start_idx: int,
        test_end_idx: int,
        horizon: int = 1,
    ) -> np.ndarray:
        """
        테스트 구간에 대해 각 시점 t에서 t+h 시점 예측값 생성
        """
        series_vals = full_series.values
        if np.isnan(series_vals).any():
            series_vals = pd.Series(series_vals).ffill().bfill().values

        y_pred = []
        c = self.params[0]
        phi = self.params[1:]

        # 슬라이딩 윈도우 기반 recursive 다중스텝 예측
        for t in range(test_start_idx, test_end_idx):
            # 과거 p_lags개 관측치 추출
            history = list(series_vals[t - self.p_lags : t])
            # h 스텝 재귀 예측
            curr_pred = 0.0
            for step in range(horizon):
                # x_{t+step} = c + sum(phi_i * x_{t+step-i})
                lags_input = np.array(history[-self.p_lags :][::-1])
                curr_pred = c + np.dot(phi, lags_input)
                curr_pred = max(0.0, float(curr_pred))  # 물리적 비음수
                history.append(curr_pred)

            y_pred.append(curr_pred)

        y_pred = np.clip(np.array(y_pred), 0.0, self.rated_capacity_mwh)
        return y_pred
