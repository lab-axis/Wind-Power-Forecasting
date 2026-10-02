"""
풍력 발전량 시계열 예측 성능 종합 평가 메트릭 모듈
(Jeju Sangmyeong Wind Power Generation Forecasting Comprehensive Metric Suite)

학술 논문(IEEE Trans. on Sustainable Energy, Applied Energy, NeurIPS/ICLR) 및 
전력 계통 실무(KPX 제주 전력시장 급전 운영)를 아우르는 5대 핵심 지표군 완비:

1. 전통적 절대 및 상대 점예측 지표:
   - MAE, RMSE, MSE
   - nMAE (%), nRMSE (%) [정격 21 MW 기준 용량 정규화 오차]
   - RSE, RRSE, RAE [시계열 벤치마크 표준]
   - WAPE (%), sMAPE (%) [체적 가중 및 대칭 백분율 오차]
   - R², Pearson Corr
   - MBE (편향), TIC (Theil's U 불평등 계수)

2. 동적 급변 및 변동성 추종 지표 (Dynamic Ramp & Volatility):
   - MAE_ramp: 1-step 시간 차분(변화율) 절대 오차 |Delta y - Delta y_hat|
   - RMSE_ramp: 램프 변화율 RMSE
   - Corr_ramp: 실제 램프와 예측 램프 간 Pearson 상관계수
   - VPR (%): 분산 보존율 (std(y_pred) / std(y_true) * 100)

3. 물리적 유계 및 무발전 상태 지표 (Physics & Boundary):
   - Bound Violation (%): 물리적 허용 구간 [0, 21.0 MWh] 위반율
   - Integrated Negative MWh: 음수 발전량 총 적분량 (물리적 모순 크기)
   - Zero F1-score & Zero Balanced Accuracy: 영발전(0 MWh) 판별 성능
   - Cut-in MAE: 시동 풍속 임계 영역(2.5 ~ 4.5 m/s) 내 국소 예측 오차

4. 전력 시장 경제성 및 비대칭 페널티 (Market & Economics):
   - Imbalance Cost (MWh): 과소예측(1.5배)과 과대예측(1.0배) 비대칭 페널티 손실

5. 보조 백분율 및 확률적 불확실성 지표 (Probabilistic & Supplementary):
   - Non-zero MAPE (%): 실제 발전량 > 0 구간에 한정한 백분율 오차 (0 나눗셈 왜곡 방지)
   - CRPS: Continuous Ranked Probability Score (확률분포 예측 품질)
   - PICP (%): 90% 신뢰구간 포함율 (Prediction Interval Coverage Probability)
   - PINAW (%): 90% 신뢰구간 정규화 평균 폭 (Sharpness)
   - Winkler Score: 신뢰구간 폭 및 이탈 페널티 종합 점수
"""

from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from scipy.stats import beta as beta_dist


# =========================================================================
# 1. 동적 급변 및 변동성 추종 지표 (Dynamic Ramp & Volatility)
# =========================================================================

def compute_ramp_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    1-step 램프 변화율(Delta y_t = y_t - y_{t-1})에 대한 추종력 및 분산 보존율 계산.
    
    장기 지평(+12h, +24h)에서 기존 신경망이 분산을 0으로 죽이고 중앙값 수평선(Flatline)으로
    엎드리는 분산 수축(Variance Collapse) 현상을 진단하는 핵심 킬러 지표입니다.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if len(yt) < 2 or len(yp) < 2:
        return {
            "mae_ramp": np.nan,
            "rmse_ramp": np.nan,
            "corr_ramp": np.nan,
            "vpr_pct": np.nan,
        }

    # 1-step 차분 벡터 (Ramp Vector)
    ramp_true = np.diff(yt)
    ramp_pred = np.diff(yp)

    mae_ramp = float(np.mean(np.abs(ramp_true - ramp_pred)))
    rmse_ramp = float(np.sqrt(np.mean((ramp_true - ramp_pred) ** 2)))

    # 램프 상관계수
    std_rt = np.std(ramp_true)
    std_rp = np.std(ramp_pred)
    if std_rt > eps and std_rp > eps:
        corr_ramp = float(np.corrcoef(ramp_true, ramp_pred)[0, 1])
    else:
        corr_ramp = 0.0

    # 분산 보존율 (Variance Preservation Ratio: VPR %)
    std_yt = np.std(yt)
    std_yp = np.std(yp)
    vpr_pct = float((std_yp / (std_yt + eps)) * 100.0)

    return {
        "mae_ramp": round(mae_ramp, 4),
        "rmse_ramp": round(rmse_ramp, 4),
        "corr_ramp": round(corr_ramp, 4),
        "vpr_pct": round(vpr_pct, 2),
    }


# =========================================================================
# 2. 물리적 유계 및 적분 음수 페널티 (Physical Boundary & Penalty)
# =========================================================================

def compute_physical_boundary_metrics(
    y_pred_raw: np.ndarray,
    rated_capacity_mwh: float = 21.0,
) -> Dict[str, float]:
    """
    클리핑 전 순수 예측값(Unclipped Raw Predictions)의 물리적 유계 위반 통계.
    - 음수 위반율 및 상한(21 MW) 초과율
    - 적분 음수 발전량(Integrated Negative Energy: 음수로 예측된 총 MWh 합)
    """
    yp = np.asarray(y_pred_raw, dtype=float).ravel()

    neg_mask = yp < 0.0
    cap_mask = yp > rated_capacity_mwh
    viol_mask = neg_mask | cap_mask

    neg_viol_count = int(np.sum(neg_mask))
    cap_viol_count = int(np.sum(cap_mask))
    bound_viol_pct = float(np.mean(viol_mask) * 100.0) if len(yp) > 0 else 0.0

    # 음수로 예측된 총 MWh 에너지 적분량
    integrated_negative_mwh = float(np.sum(np.maximum(0.0, -yp)))

    min_pred = float(np.min(yp)) if len(yp) > 0 else 0.0
    max_pred = float(np.max(yp)) if len(yp) > 0 else 0.0

    return {
        "bound_violation_pct": round(bound_viol_pct, 4),
        "neg_viol_count": neg_viol_count,
        "cap_viol_count": cap_viol_count,
        "integrated_negative_mwh": round(integrated_negative_mwh, 4),
        "min_pred_mwh": round(min_pred, 4),
        "max_pred_mwh": round(max_pred, 4),
    }


# =========================================================================
# 3. 영발전(0 MWh) 및 시동 풍속 임계치 지표 (Zero-State & Cut-in Physics)
# =========================================================================

def compute_zero_state_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-4,
) -> Dict[str, float]:
    """
    구조적 영발전(Zero-Inflation, 약 17.5%)에 대한 이진 판별 정확도 평가.
    실측값 및 예측값이 zero_threshold 이하이면 1(Zero State), 초과이면 0(Active State).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    label_true = (yt <= zero_threshold).astype(int)
    label_pred = (yp <= zero_threshold).astype(int)

    if len(label_true) == 0 or len(np.unique(label_true)) < 2:
        return {
            "zero_f1": np.nan,
            "zero_balanced_acc": np.nan,
            "zero_precision": np.nan,
            "zero_recall": np.nan,
        }

    f1 = float(f1_score(label_true, label_pred, zero_division=0))
    b_acc = float(balanced_accuracy_score(label_true, label_pred))
    prec = float(precision_score(label_true, label_pred, zero_division=0))
    rec = float(recall_score(label_true, label_pred, zero_division=0))

    return {
        "zero_f1": round(f1, 4),
        "zero_balanced_acc": round(b_acc, 4),
        "zero_precision": round(prec, 4),
        "zero_recall": round(rec, 4),
    }


def compute_cut_in_transition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wind_speed: np.ndarray,
    cut_in_range: Tuple[float, float] = (2.5, 4.5),
) -> Dict[str, float]:
    """
    풍력 터빈이 기동할지 정지할지 가장 불확실성이 높은 시동 풍속 인접 영역(2.5 ~ 4.5 m/s)의
    국소 예측 오차(MAE, RMSE)를 계산합니다.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ws = np.asarray(wind_speed, dtype=float).ravel()

    mask = (ws >= cut_in_range[0]) & (ws <= cut_in_range[1])
    if np.sum(mask) == 0:
        return {"mae_cut_in": np.nan, "rmse_cut_in": np.nan, "cut_in_count": 0}

    mae_ci = float(np.mean(np.abs(yt[mask] - yp[mask])))
    rmse_ci = float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))

    return {
        "mae_cut_in": round(mae_ci, 4),
        "rmse_cut_in": round(rmse_ci, 4),
        "cut_in_count": int(np.sum(mask)),
    }


# =========================================================================
# 4. 전력 계통 시장 경제성 지표 (Market Asymmetric Imbalance Loss)
# =========================================================================

def compute_market_imbalance_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_under: float = 1.5,
    c_over: float = 1.0,
) -> Dict[str, float]:
    """
    전력 계통의 비대칭 불평형 정산금(Grid Imbalance Cost) 모사:
    - 과소예측(y > y_hat): 급전 예비력 추가 기동 비용 (c_under = 1.5)
    - 과대예측(y < y_hat): 신재생 출력제어 및 잉여 페널티 (c_over = 1.0)
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    under_err = np.maximum(0.0, yt - yp)
    over_err = np.maximum(0.0, yp - yt)

    imbalance_cost = float(np.mean(c_under * under_err + c_over * over_err))
    mean_under = float(np.mean(under_err))
    mean_over = float(np.mean(over_err))

    return {
        "imbalance_loss_mwh": round(imbalance_cost, 4),
        "under_forecast_mwh": round(mean_under, 4),
        "over_forecast_mwh": round(mean_over, 4),
    }


# =========================================================================
# 5. 전통 보조 지표 (Non-zero MAPE)
# =========================================================================

def compute_nonzero_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-4,
) -> float:
    """
    실제 발전량이 존재하는 유효 가동 구간(y_true > zero_threshold)에 한정하여
    계산한 MAPE (0 나눗셈으로 인한 무한대 왜곡 배제).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    pos_mask = yt > zero_threshold
    if np.sum(pos_mask) == 0:
        return np.nan

    mape_val = float(np.mean(np.abs(yt[pos_mask] - yp[pos_mask]) / yt[pos_mask]) * 100.0)
    return round(mape_val, 2)


# =========================================================================
# 6. 확률적 불확실성 지표 (CRPS & Prediction Intervals)
# =========================================================================

def compute_crps_point(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    단일 점예측 모델의 CRPS (Continuous Ranked Probability Score):
    Dirac delta 예측 분포에 대한 CRPS는 정확히 MAE와 일치함.
    CRPS(Dirac(y_hat), y) = |y - y_hat|.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return round(float(np.mean(np.abs(yt - yp))), 4)


def compute_crps_hurdle_beta(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    alpha: np.ndarray,
    beta_param: np.ndarray,
    rated_capacity_mwh: float = 21.0,
    n_eval_points: int = 101,
) -> float:
    """
    Bernoulli-Beta Hurdle 모델의 엄밀한 수치 적분 CRPS (Gneiting & Raftery 2007):
    CRPS(F, y) = integral_0^C (F(z) - I(z >= y))^2 dz
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    p_p = np.asarray(p_pos, dtype=float).ravel()
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta_param, dtype=float).ravel()

    C = rated_capacity_mwh
    y_norm = np.clip(yt / C, 0.0, 1.0)

    # 101개 등간격 정규화 그리드 [0, 1]
    grid = np.linspace(0.0, 1.0, n_eval_points)
    z = grid[np.newaxis, :]  # [1, G]

    # Hurdle CDF: F(z) = (1 - p_pos) + p_pos * Beta_CDF(z; alpha, beta)
    p_zero = (1.0 - p_p)[:, np.newaxis]  # [N, 1]
    p_active = p_p[:, np.newaxis]        # [N, 1]
    a_mat = a[:, np.newaxis]             # [N, 1]
    b_mat = b[:, np.newaxis]             # [N, 1]

    cdf_beta = beta_dist.cdf(z, a_mat, b_mat)
    F_z = p_zero + p_active * cdf_beta  # [N, G]

    # Heaviside Step Function H(z - y) = I(z >= y)
    H_z = (z >= y_norm[:, np.newaxis]).astype(float)  # [N, G]

    diff_sq = (F_z - H_z) ** 2

    # 수치 사다리꼴 적분 (NumPy 2.0+ 호환)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    crps_per_sample = C * trap_fn(diff_sq, grid, axis=1)

    return round(float(np.mean(crps_per_sample)), 4)


def hurdle_beta_quantiles(
    p_pos: np.ndarray,
    alpha: np.ndarray,
    beta_param: np.ndarray,
    q: float,
    rated_capacity_mwh: float = 21.0,
) -> np.ndarray:
    """
    Bernoulli-Beta Hurdle 누적분포함수의 분위수 역함수 F^{-1}(q).
    q <= (1 - p_pos) 인 경우 정확히 0.0 MWh 출력.
    """
    p_p = np.asarray(p_pos, dtype=float).ravel()
    a = np.asarray(alpha, dtype=float).ravel()
    b = np.asarray(beta_param, dtype=float).ravel()

    p_zero = 1.0 - p_p
    target_beta_p = (q - p_zero) / np.maximum(p_p, 1e-6)

    res = np.zeros_like(p_p)
    mask_pos = target_beta_p > 0.0

    if np.any(mask_pos):
        clipped_p = np.clip(target_beta_p[mask_pos], 1e-5, 1.0 - 1e-5)
        res[mask_pos] = beta_dist.ppf(clipped_p, a[mask_pos], b[mask_pos]) * rated_capacity_mwh

    return res


def compute_prediction_interval_metrics(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    rated_capacity_mwh: float = 21.0,
    nominal_coverage: float = 0.90,
) -> Dict[str, float]:
    """
    예측 신뢰구간(Prediction Intervals) 품질 평가:
    - PICP: Prediction Interval Coverage Probability (목표: >= 90%)
    - PINAW: Normalized Average Width (선명성/Sharpness, 낮을수록 우수)
    - Winkler Score: 구간 폭 및 이탈 페널티 종합 손실 (Winkler 1972)
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    lb = np.asarray(lower_bound, dtype=float).ravel()
    ub = np.asarray(upper_bound, dtype=float).ravel()

    N = len(yt)
    if N == 0:
        return {"picp_pct": np.nan, "pinaw_pct": np.nan, "winkler_score": np.nan}

    # 1. PICP (%)
    in_interval = (yt >= lb) & (yt <= ub)
    picp_pct = float(np.mean(in_interval) * 100.0)

    # 2. PINAW (%)
    width = ub - lb
    pinaw_pct = float((np.mean(width) / rated_capacity_mwh) * 100.0)

    # 3. Winkler Score (alpha = 1 - nominal_coverage, e.g. 0.10)
    alpha_signif = 1.0 - nominal_coverage
    penalty_lower = (2.0 / alpha_signif) * np.maximum(0.0, lb - yt)
    penalty_upper = (2.0 / alpha_signif) * np.maximum(0.0, yt - ub)
    winkler = float(np.mean(width + penalty_lower + penalty_upper))

    return {
        "picp_90_pct": round(picp_pct, 2),
        "pinaw_90_pct": round(pinaw_pct, 2),
        "winkler_score_90": round(winkler, 4),
    }


# =========================================================================
# 7. 종합 마스터 평가 함수 (Master Forecast Evaluation)
# =========================================================================

def evaluate_wind_forecast(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_pred_raw: Optional[Union[np.ndarray, list]] = None,
    wind_speed: Optional[Union[np.ndarray, list]] = None,
    rated_capacity_mwh: float = 21.0,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    상명풍력 시간별 발전량(MWh) 예측 결과 종합 평가 마스터 함수.
    
    기존 14대 표준 회귀 지표를 완벽히 유지(100% 하위 호환)하면서,
    동적 램프 오차, 분산 보존율, 비대칭 불평형 손실, Non-zero MAPE,
    물리적 유계 위반 통계, 영발전 판별 F1을 일괄 산출합니다.
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    # 클리핑 전 원본 예측값 보존 (물리적 위반율 산출용)
    if y_pred_raw is not None:
        raw_pred = np.asarray(y_pred_raw, dtype=float).ravel()
    else:
        raw_pred = y_pred_arr.copy()

    # 공정한 점예측 지표 평가를 위해 물리적 범위 [0, rated_capacity_mwh] 클리핑
    y_pred_clipped = np.clip(y_pred_arr, 0.0, rated_capacity_mwh)

    # 유효값 마스킹 (NaN 배제)
    mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_clipped)
    yt = y_true_arr[mask]
    yp = y_pred_clipped[mask]
    yraw = raw_pred[mask]

    if len(yt) == 0:
        return {
            "mae": np.nan, "rmse": np.nan, "mse": np.nan,
            "nmae_pct": np.nan, "nrmse_pct": np.nan,
            "rse": np.nan, "rrse": np.nan, "rae": np.nan,
            "wape_pct": np.nan, "smape_pct": np.nan,
            "r2": np.nan, "corr": np.nan, "mbe": np.nan, "tic": np.nan,
            "count": 0,
        }

    # 1. 절대 오차 (Absolute Errors)
    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))

    # 2. 정규화 오차 (Normalized Errors - 정격용량 21MW 기준)
    nmae_pct = float((mae / rated_capacity_mwh) * 100.0)
    nrmse_pct = float((rmse / rated_capacity_mwh) * 100.0)

    # 3. 상대 오차 (Relative Errors - Informer, PatchTST 등 시계열 논문 표준)
    y_mean = np.mean(yt)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - y_mean) ** 2)
    sa_res = np.sum(np.abs(yt - yp))
    sa_tot = np.sum(np.abs(yt - y_mean))

    rse = float(ss_res / (ss_tot + eps))
    rrse = float(np.sqrt(rse))
    rae = float(sa_res / (sa_tot + eps))

    # 4. 백분율 및 산업 표준 (WAPE, sMAPE)
    sum_yt = np.sum(yt)
    wape_pct = float((sa_res / (sum_yt + eps)) * 100.0)

    denominator = np.abs(yt) + np.abs(yp) + eps
    smape_pct = float(np.mean(200.0 * np.abs(yt - yp) / denominator))

    # 5. 상관관계 및 결정계수 (R2, Pearson Corr)
    r2 = float(r2_score(yt, yp))
    if np.std(yt) > eps and np.std(yp) > eps:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = 0.0

    # 6. 방향성 및 불평등 계수 (MBE, TIC)
    mbe = float(np.mean(yp - yt))
    numerator_tic = np.sqrt(mse)
    denominator_tic = np.sqrt(np.mean(yt ** 2)) + np.sqrt(np.mean(yp ** 2)) + eps
    tic = float(numerator_tic / denominator_tic)

    # 7. 동적 램프 및 분산 보존 지표
    ramp_m = compute_ramp_metrics(yt, yp, eps=eps)

    # 8. 물리적 유계 통계 (클리핑 전 원본 기준)
    bound_m = compute_physical_boundary_metrics(yraw, rated_capacity_mwh=rated_capacity_mwh)

    # 9. 비대칭 전력시장 불평형 손실
    mkt_m = compute_market_imbalance_cost(yt, yp, c_under=1.5, c_over=1.0)

    # 10. Non-zero MAPE
    mape_nz = compute_nonzero_mape(yt, yp, zero_threshold=1e-4)

    # 11. 영발전 F1-score & 균형 정확도
    zero_m = compute_zero_state_metrics(yt, yp, zero_threshold=1e-4)

    # 기본 결과 딕셔너리 조합
    out: Dict[str, Any] = {
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
        "mape_nonzero_pct": mape_nz,
        "r2": round(r2, 4),
        "corr": round(corr, 4),
        "mbe": round(mbe, 4),
        "tic": round(tic, 4),
        "count": int(len(yt)),
        # Dynamic Ramp & Variance
        "mae_ramp": ramp_m["mae_ramp"],
        "rmse_ramp": ramp_m["rmse_ramp"],
        "corr_ramp": ramp_m["corr_ramp"],
        "vpr_pct": ramp_m["vpr_pct"],
        # Physical Bounds
        "bound_violation_pct": bound_m["bound_violation_pct"],
        "integrated_negative_mwh": bound_m["integrated_negative_mwh"],
        "min_pred_mwh": bound_m["min_pred_mwh"],
        "max_pred_mwh": bound_m["max_pred_mwh"],
        # Zero State
        "zero_f1": zero_m["zero_f1"],
        "zero_balanced_acc": zero_m["zero_balanced_acc"],
        # Market Imbalance
        "imbalance_loss_mwh": mkt_m["imbalance_loss_mwh"],
    }

    # 만약 풍속 데이터가 제공된 경우 Cut-in 영역 평가 추가
    if wind_speed is not None:
        ws_arr = np.asarray(wind_speed, dtype=float).ravel()[mask]
        ci_m = compute_cut_in_transition_metrics(yt, yp, ws_arr)
        out["mae_cut_in"] = ci_m["mae_cut_in"]
        out["rmse_cut_in"] = ci_m["rmse_cut_in"]

    return out


__all__ = [
    "evaluate_wind_forecast",
    "compute_ramp_metrics",
    "compute_physical_boundary_metrics",
    "compute_zero_state_metrics",
    "compute_cut_in_transition_metrics",
    "compute_market_imbalance_cost",
    "compute_nonzero_mape",
    "compute_crps_point",
    "compute_crps_hurdle_beta",
    "hurdle_beta_quantiles",
    "compute_prediction_interval_metrics",
]
