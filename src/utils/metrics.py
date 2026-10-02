"""
풍력 발전량 시계열 예측 성능 종합 평가 메트릭 모듈
- 학술 논문 및 실무에서 널리 인정받는 표준 시계열/에너지 예측 지표군 완비
- 절대 오차: MAE, RMSE, MSE
- 정규화 오차 (설비용량 21 MW 기준): nMAE (%), nRMSE (%)
- 상대 오차 (시계열 벤치마크 표준): RSE, RRSE, RAE
- 백분율 및 산업 표준: WAPE (%), sMAPE (%)
- 상관관계 및 적합도: R², Pearson Corr
- 방향성 및 통계량: MBE (편향), TIC (Theil's U 계수)
"""

import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_wind_forecast(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    rated_capacity_mwh: float = 21.0,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    상명풍력 시간별 발전량(MWh) 예측 결과 종합 평가

    Args:
        y_true: 실제 발전량 (MWh)
        y_pred: 예측 발전량 (MWh)
        rated_capacity_mwh: 상명풍력 설비용량 (21.0 MW -> 1시간 최대 21.0 MWh)
        eps: 분모 0 방지용 엡실론

    Returns:
        dict:
          - mae, rmse, mse
          - nmae_pct, nrmse_pct
          - rse, rrse, rae
          - wape_pct, smape_pct
          - r2, corr
          - mbe, tic
          - count
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # 음수 예측값은 물리적으로 0으로 클리핑 (풍력 발전량은 음수 불가)
    y_pred = np.clip(y_pred, 0.0, rated_capacity_mwh)

    # 유효값 마스킹 (NaN 배제)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mse": np.nan,
            "nmae_pct": np.nan,
            "nrmse_pct": np.nan,
            "rse": np.nan,
            "rrse": np.nan,
            "rae": np.nan,
            "wape_pct": np.nan,
            "smape_pct": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "mbe": np.nan,
            "tic": np.nan,
            "count": 0,
        }

    # 1. 절대 오차 (Absolute Errors)
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))

    # 2. 정규화 오차 (Normalized Errors - 정격용량 21MW 기준)
    nmae_pct = float((mae / rated_capacity_mwh) * 100.0)
    nrmse_pct = float((rmse / rated_capacity_mwh) * 100.0)

    # 3. 상대 오차 (Relative Errors - Informer, PatchTST 등 시계열 논문 표준)
    # RSE: sum((y - y_hat)^2) / sum((y - y_bar)^2)
    # RRSE: sqrt(RSE)
    # RAE: sum(|y - y_hat|) / sum(|y - y_bar|)
    y_mean = np.mean(y_t)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_mean) ** 2)
    sa_res = np.sum(np.abs(y_t - y_p))
    sa_tot = np.sum(np.abs(y_t - y_mean))

    rse = float(ss_res / (ss_tot + eps))
    rrse = float(np.sqrt(rse))
    rae = float(sa_res / (sa_tot + eps))

    # 4. 백분율 및 산업 표준 (WAPE, sMAPE)
    # WAPE: sum(|y - y_hat|) / sum(y) * 100
    sum_yt = np.sum(y_t)
    wape_pct = float((sa_res / (sum_yt + eps)) * 100.0)

    # sMAPE: 200 * mean(|y - y_hat| / (|y| + |y_hat| + eps))
    denominator = np.abs(y_t) + np.abs(y_p) + eps
    smape_pct = float(np.mean(200.0 * np.abs(y_t - y_p) / denominator))

    # 5. 상관관계 및 결정계수 (R2, Pearson Corr)
    r2 = float(r2_score(y_t, y_p))

    # Pearson Correlation Coefficient (r)
    if np.std(y_t) > 0 and np.std(y_p) > 0:
        corr = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        corr = 0.0

    # 6. 방향성 및 불평등 계수 (MBE, TIC)
    # MBE: mean(y_hat - y) (양수: 과대전망, 음수: 과소전망)
    mbe = float(np.mean(y_p - y_t))

    # TIC (Theil's Inequality Coefficient): sqrt(mean((y - y_hat)^2)) / (sqrt(mean(y^2)) + sqrt(mean(y_hat^2)))
    numerator_tic = np.sqrt(mse)
    denominator_tic = np.sqrt(np.mean(y_t ** 2)) + np.sqrt(np.mean(y_p ** 2)) + eps
    tic = float(numerator_tic / denominator_tic)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mse": round(mse, 4),
        "nmae_pct": round(nmae_pct, 4),
        "nrmse_pct": round(nrmse_pct, 4),
        "rse": round(rse, 4),
        "rrse": round(rrse, 4),
        "rae": round(rae, 4),
        "wape_pct": round(wape_pct, 4),
        "smape_pct": round(smape_pct, 4),
        "r2": round(r2, 4),
        "corr": round(corr, 4),
        "mbe": round(mbe, 4),
        "tic": round(tic, 4),
        "count": int(len(y_t)),
    }
