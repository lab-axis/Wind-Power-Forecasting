"""
LightGBM 기반 상명풍력 시간별 발전량(MWh) 다중 Horizon 예측 모델 모듈
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.features.lag_features import create_lag_features, create_rolling_features, create_horizon_target
from src.features.time_features import add_cyclical_time_features
from src.utils.metrics import evaluate_wind_forecast


class SangmyeongLightGBMForecaster:
    """
    상명풍력 발전량 예측을 위한 LightGBM 다중 Horizon 모델
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        target_col: str = "generation_mwh",
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
        rated_capacity_mwh: float = 21.0,
    ):
        if horizons is None:
            horizons = [1, 3, 6, 12, 24]
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24, 48, 168]
        if windows is None:
            windows = [3, 6, 24]

        self.horizons = horizons
        self.target_col = target_col
        self.lags = lags
        self.windows = windows
        self.rated_capacity_mwh = rated_capacity_mwh
        self.models: Dict[int, lgb.Booster] = {}
        self.feature_cols: List[str] = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터셋에 lag, rolling, time 주기 피처 생성
        """
        df_feat = df.copy()
        df_feat["datetime"] = pd.to_datetime(df_feat["datetime"])
        df_feat.sort_values("datetime", inplace=True)

        # 1. 시간 주기 피처
        df_feat = add_cyclical_time_features(df_feat, time_col="datetime")

        # 2. 발전량 lag 피처
        df_feat = create_lag_features(df_feat, target_cols=[self.target_col], lags=self.lags)

        # 3. 발전량 rolling 피처
        df_feat = create_rolling_features(df_feat, target_cols=[self.target_col], windows=self.windows)

        # 사용 가능한 피처 컬럼 식별
        exclude = {
            "datetime",
            "base_date",
            "hour",
            "raw_generation",
            "scale_unit_assumed",
            "is_capacity_exceeded",
            "is_negative",
            "is_zero",
        }
        self.feature_cols = [c for c in df_feat.columns if c not in exclude and not c.startswith("target_h")]
        return df_feat

    def train_horizon(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        horizon: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> lgb.Booster:
        """
        특정 horizon에 대해 LightGBM 모델 학습
        """
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }

        # 타겟: t+h 시점 발전량
        y_train = create_horizon_target(df_train, target_col=self.target_col, horizon=horizon)
        y_val = create_horizon_target(df_val, target_col=self.target_col, horizon=horizon)

        X_train = df_train[self.feature_cols]
        X_val = df_val[self.feature_cols]

        # 결측 타겟 제거
        train_mask = y_train.notna() & X_train.notna().all(axis=1)
        val_mask = y_val.notna() & X_val.notna().all(axis=1)

        dtrain = lgb.Dataset(X_train[train_mask], label=y_train[train_mask])
        dval = lgb.Dataset(X_val[val_mask], label=y_val[val_mask], reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        booster = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            callbacks=callbacks,
        )

        self.models[horizon] = booster
        return booster

    def evaluate(
        self, df_test: pd.DataFrame, horizon: int
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        특정 horizon에 대해 테스트 데이터셋 평가
        """
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon}에 대해 학습된 모델이 없습니다.")

        booster = self.models[horizon]
        y_true = create_horizon_target(df_test, target_col=self.target_col, horizon=horizon)
        X_test = df_test[self.feature_cols]

        valid_mask = y_true.notna() & X_test.notna().all(axis=1)
        X_eval = X_test[valid_mask]
        y_eval = y_true[valid_mask].values

        y_pred = booster.predict(X_eval)
        # Preserve raw predictions; point metrics are clipped centrally.
        self.last_eval_indices = np.flatnonzero(valid_mask.to_numpy())

        metrics = evaluate_wind_forecast(
            y_eval, y_pred, rated_capacity_mwh=self.rated_capacity_mwh
        )
        metrics["horizon_hours"] = horizon

        return metrics, y_eval, y_pred
