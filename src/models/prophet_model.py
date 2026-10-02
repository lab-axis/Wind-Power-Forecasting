"""
Prophet 기반 시계열 예측 모델 모듈
- Taylor & Letham (2018) "Forecasting at Scale" (Facebook Prophet)
- 비선형 추세(Trend) + 일간(Daily)/주간(Weekly)/연간(Yearly) 복합 계절성 분해 모델
- 재생에너지 및 풍력 발전 기저 트렌드 예측 대표 모델
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from prophet import Prophet
from src.utils.metrics import evaluate_wind_forecast


class ProphetForecaster:
    """
    상명풍력 시간별 발전량(generation_mwh) 예측을 위한 Prophet 모델
    """

    def __init__(
        self,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        rated_capacity_mwh: float = 21.0,
    ):
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.rated_capacity_mwh = rated_capacity_mwh
        self.model = None

    def fit(
        self,
        df: pd.DataFrame,
        time_col: str = "datetime",
        target_col: str = "generation_mwh",
        exog_cols: Optional[List[str]] = None,
    ):
        """
        Prophet 포맷 (ds, y)으로 변환 후 모델 적합
        """
        pdf = pd.DataFrame()
        pdf["ds"] = pd.to_datetime(df[time_col])
        pdf["y"] = df[target_col].values

        self.model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        if exog_cols:
            for col in exog_cols:
                pdf[col] = df[col].values
                self.model.add_regressor(col)

        self.model.fit(pdf)
        return self

    def predict(
        self,
        future_df: pd.DataFrame,
        time_col: str = "datetime",
        exog_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        미래 데이터프레임에 대한 발전량 예측 수행
        """
        pdf = pd.DataFrame()
        pdf["ds"] = pd.to_datetime(future_df[time_col])
        if exog_cols:
            for col in exog_cols:
                pdf[col] = future_df[col].values

        forecast = self.model.predict(pdf)
        preds = forecast["yhat"].values
        # 물리적 상한 클리핑 [0.0, 21.0]
        return np.clip(preds, 0.0, self.rated_capacity_mwh)
