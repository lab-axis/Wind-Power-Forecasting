"""
시간 주기성 인코딩(Cyclical Time Features) 모듈
- README_sangmyeong_wind_generation_forecasting.md 섹션 22 기준
- 임의의 계절 범주(봄/여름/가을/겨울) 대신 연속적인 sin/cos 삼각함수 인코딩 적용
"""

import pandas as pd
import numpy as np


def add_cyclical_time_features(
    df: pd.DataFrame, time_col: str = "datetime"
) -> pd.DataFrame:
    """
    시간대(0~23) 및 연중일(1~365/366)에 대한 삼각함수 주기 인코딩 피처 추가:
    - hour_sin, hour_cos (24시간 주기)
    - day_of_year_sin, day_of_year_cos (연간 주기)
    - month_sin, month_cos (12개월 주기)
    """
    df_out = df.copy()
    dt = pd.to_datetime(df_out[time_col])

    # 1. 일중 주기 (24시간)
    hour = dt.dt.hour
    df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # 2. 연중 주기 (365.25일)
    day_of_year = dt.dt.dayofyear
    df_out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 3. 월 주기 (12개월)
    month = dt.dt.month
    df_out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return df_out
