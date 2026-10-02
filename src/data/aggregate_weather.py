"""
기상 관측 데이터 1시간 집계(Resampling & Aggregation) 모듈
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.wind_direction import wind_direction_to_sin_cos, sin_cos_to_wind_direction


def aggregate_weather_to_hourly(
    df_raw: pd.DataFrame,
    time_col: str = "datetime",
    temp_col: str = "temperature",
    wind_spd_col: str = "wind_speed",
    wind_dir_col: str = "wind_direction",
    humidity_col: str = "humidity",
    pressure_col: str = "pressure",
    precipitation_col: str = "precipitation",
) -> pd.DataFrame:
    """
    고해상도(1분, 10분 등) 기상 시계열을 1시간 단위로 표준 집계
    집계 원칙:
    - temperature, wind_speed, humidity, pressure: mean
    - wind_direction: sin/cos 변환 후 mean -> wind_dir_sin, wind_dir_cos 생성
    - precipitation: sum
    """
    df = df_raw.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # 풍향 sin/cos 성분 생성 후 집계
    if wind_dir_col in df.columns:
        sin_v, cos_v = wind_direction_to_sin_cos(df[wind_dir_col])
        df["wind_dir_sin"] = sin_v
        df["wind_dir_cos"] = cos_v

    agg_dict = {}
    if temp_col in df.columns:
        agg_dict[temp_col] = "mean"
    if wind_spd_col in df.columns:
        agg_dict[wind_spd_col] = "mean"
    if "wind_dir_sin" in df.columns:
        agg_dict["wind_dir_sin"] = "mean"
    if "wind_dir_cos" in df.columns:
        agg_dict["wind_dir_cos"] = "mean"
    if humidity_col in df.columns:
        agg_dict[humidity_col] = "mean"
    if pressure_col in df.columns:
        agg_dict[pressure_col] = "mean"
    if precipitation_col in df.columns:
        agg_dict[precipitation_col] = "sum"

    df_hourly = df.resample("1h").agg(agg_dict).reset_index()

    # 원형 평균 풍향 복원 (참고용 컬럼)
    if "wind_dir_sin" in df_hourly.columns and "wind_dir_cos" in df_hourly.columns:
        df_hourly["wind_direction_mean"] = sin_cos_to_wind_direction(
            df_hourly["wind_dir_sin"], df_hourly["wind_dir_cos"]
        )

    return df_hourly
