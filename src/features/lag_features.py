"""
시계열 Lagged 피처 및 Rolling 통계 생성 모듈
- README_sangmyeong_wind_generation_forecasting.md 섹션 28 기준
- 상명풍력 시간별 발전량(generation_mwh) 및 기상 변수에 대한 과거 lag 생성
- 이동평균(rolling mean), 이동표준편차(rolling std) 생성
- 미래 데이터 누수(Data Leakage) 원천 차단 (현재 예측 시점 t 이전만 참조)
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def create_lag_features(
    df: pd.DataFrame,
    target_cols: List[str],
    lags: Optional[List[int]] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 과거 lag 시점의 피처를 생성합니다.
    기본 lags: [1, 2, 3, 6, 12, 24, 48, 168] (1시간, 2시간, 3시간, 6시간, 12시간, 24시간, 48시간, 7일 전)
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]

    df_out = df.copy()

    for col in target_cols:
        if col not in df_out.columns:
            continue
        for lag in lags:
            new_col_name = f"{col}_lag_{lag}"
            if group_col and group_col in df_out.columns:
                df_out[new_col_name] = df_out.groupby(group_col)[col].shift(lag)
            else:
                df_out[new_col_name] = df_out[col].shift(lag)

    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    target_cols: List[str],
    windows: Optional[List[int]] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    지정된 컬럼들에 대해 과거 윈도우 이동평균 및 이동표준편차를 생성합니다.
    데이터 누수 방지를 위해 시점 t-1 이전의 관측치만 윈도우 계산에 사용합니다 (shift(1) 적용).
    기본 windows: [3, 6, 24]
    """
    if windows is None:
        windows = [3, 6, 24]

    df_out = df.copy()

    for col in target_cols:
        if col not in df_out.columns:
            continue
        for win in windows:
            mean_col = f"{col}_roll_mean_{win}"
            std_col = f"{col}_roll_std_{win}"

            if group_col and group_col in df_out.columns:
                shifted = df_out.groupby(group_col)[col].shift(1)
                df_out[mean_col] = (
                    shifted.groupby(df_out[group_col])
                    .rolling(win, min_periods=1)
                    .mean()
                    .reset_index(drop=True)
                )
                df_out[std_col] = (
                    shifted.groupby(df_out[group_col])
                    .rolling(win, min_periods=1)
                    .std()
                    .reset_index(drop=True)
                )
            else:
                shifted = df_out[col].shift(1)
                df_out[mean_col] = shifted.rolling(win, min_periods=1).mean()
                df_out[std_col] = shifted.rolling(win, min_periods=1).std()

    return df_out


def create_horizon_target(
    df: pd.DataFrame,
    target_col: str = "generation_mwh",
    horizon: int = 1,
) -> pd.Series:
    """
    예측 horizon h 시간 앞선 타겟 생성 (y_{t+h} = shift(-h))
    """
    return df[target_col].shift(-horizon)
