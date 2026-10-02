"""
풍향 변환 및 원형 통계(Circular Statistics) 모듈

풍향 각도(0~360도)는 불연속성(359도와 1도의 산술평균은 180도 남풍으로 왜곡됨)을 가지므로
반드시 sin, cos 분해를 통해 벡터 성분으로 변환하여 처리해야 함.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def wind_direction_to_sin_cos(
    degrees: Union[float, np.ndarray, pd.Series]
) -> Tuple[Union[float, np.ndarray, pd.Series], Union[float, np.ndarray, pd.Series]]:
    """
    풍향 각도(0° ~ 360°)를 sin 및 cos 성분으로 변환
    """
    rad = np.deg2rad(degrees)
    sin_val = np.sin(rad)
    cos_val = np.cos(rad)
    return sin_val, cos_val


def sin_cos_to_wind_direction(
    sin_val: Union[float, np.ndarray, pd.Series],
    cos_val: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    sin 및 cos 성분으로부터 평균 풍향 각도(0° ~ 360°) 복원
    """
    deg = np.rad2deg(np.arctan2(sin_val, cos_val))
    # [-180, 180] -> [0, 360] 변환
    deg = np.where(deg < 0, deg + 360.0, deg)
    return deg


def add_wind_direction_components(
    df: pd.DataFrame, direction_col: str = "wind_direction"
) -> pd.DataFrame:
    """
    DataFrame에 wind_dir_sin, wind_dir_cos 컬럼 추가
    """
    df_out = df.copy()
    if direction_col in df_out.columns:
        sin_v, cos_v = wind_direction_to_sin_cos(df_out[direction_col])
        df_out["wind_dir_sin"] = sin_v
        df_out["wind_dir_cos"] = cos_v
    return df_out


# Backward compatibility and alternate naming aliases
encode_wind_direction = wind_direction_to_sin_cos
decode_wind_direction = sin_cos_to_wind_direction

__all__ = [
    "wind_direction_to_sin_cos",
    "sin_cos_to_wind_direction",
    "add_wind_direction_components",
    "encode_wind_direction",
    "decode_wind_direction",
]
