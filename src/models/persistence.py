"""
Baseline Persistence 예측 모델 모듈
- Naive Persistence: t 시점 발전량을 그대로 t+h 시점 예측값으로 사용 (y_hat_{t+h} = y_t)
- 24h Diurnal Persistence: 24시간 전 동일 시점 발전량을 예측값으로 사용 (y_hat_{t+h} = y_{t+h-24})
"""

import numpy as np
import pandas as pd
from typing import Optional


class NaivePersistence:
    """단순 직전값 지속(Persistence) 모델"""
    def __init__(self):
        pass

    def predict(self, series: pd.Series, horizon: int = 1) -> pd.Series:
        """
        시점 t에서의 값으로 시점 t+h를 예측:
        t 시점의 예측값 = series.shift(0)
        t+h 시점과 정렬하기 위해 series.shift(horizon)을 반환
        """
        return series.shift(horizon)


class DiurnalPersistence:
    """24시간 주기(동일 시각) 지속 모델"""
    def __init__(self, period: int = 24):
        self.period = period

    def predict(self, series: pd.Series, horizon: int = 1) -> pd.Series:
        """
        t+h 시점의 예측값을 t+h - 24 시점의 관측값으로 사용:
        예: horizon=1일 때 t+1 시점의 예측값 = t-23 시점 관측치
        """
        shift_amount = 24
        return series.shift(shift_amount)
