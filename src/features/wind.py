"""
풍향 및 풍속 관련 피처 변환 유틸리티
- 풍향 각도(degree, 0~360)를 sin/cos 직교 좌표 성분으로 변환
- 원형 데이터(circular data)의 왜곡 없는 통계량 계산 및 집계 지원
"""

import numpy as np
import pandas as pd


def encode_wind_direction(
    deg_series: pd.Series | np.ndarray,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """
    풍향 각도(0~360도)를 sin, cos 성분으로 변환합니다.

    Args:
        deg_series: 풍향 각도 (단위: degree)

    Returns:
        (wind_dir_sin, wind_dir_cos)
    """
    rad = np.deg2rad(deg_series)
    return np.sin(rad), np.cos(rad)


def decode_wind_direction(
    sin_series: pd.Series | np.ndarray,
    cos_series: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """
    sin, cos 성분으로부터 풍향 각도(0~360도)를 복원합니다.

    Args:
        sin_series: 풍향 sin 성분
        cos_series: 풍향 cos 성분

    Returns:
        풍향 각도 (단위: degree, 0 <= deg < 360)
    """
    rad = np.arctan2(sin_series, cos_series)
    deg = np.rad2deg(rad)
    return (deg + 360.0) % 360.0
