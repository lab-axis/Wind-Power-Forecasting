"""
시각화 및 플롯 유틸리티 모듈
- Windows 한글 폰트(맑은 고딕) 자동 설정 및 마이너스 부호 깨짐 방지
- 시계열 예측 결과 비교 플롯, Horizon별 성능 곡선 플롯, 상관관계 히트맵 등
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_korean_font():
    """Windows 환경 한글 폰트 및 마이너스 폰트 깨짐 방지 설정"""
    plt.rc("font", family="Malgun Gothic")
    plt.rc("axes", unicode_minus=False)


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_index: pd.DatetimeIndex | None = None,
    horizon_label: str = "+1h",
    save_path: str | None = None,
    num_samples: int = 400,
    rated_capacity_mwh: float = 21.0,
):
    """
    상명풍력 실제 발전량(MWh)과 모델 예측값을 비교하는 시계열 플롯을 생성합니다.
    """
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(14, 5))

    plot_slice = slice(-num_samples, None)
    yt = y_true[plot_slice]
    yp = y_pred[plot_slice]

    if time_index is not None:
        t = time_index[plot_slice]
        ax.plot(t, yt, label="Actual Generation (MWh)", color="black", alpha=0.8, linewidth=1.2)
        ax.plot(t, yp, label=f"Predicted ({horizon_label})", color="dodgerblue", linestyle="--", linewidth=1.5)
    else:
        ax.plot(yt, label="Actual Generation (MWh)", color="black", alpha=0.8, linewidth=1.2)
        ax.plot(yp, label=f"Predicted ({horizon_label})", color="dodgerblue", linestyle="--", linewidth=1.5)

    ax.axhline(rated_capacity_mwh, color="gray", linestyle=":", label=f"Rated Capacity ({rated_capacity_mwh} MW)")
    ax.set_title(f"상명풍력발전소 시간별 발전량 예측 결과 - {horizon_label}", fontsize=13, pad=10)
    ax.set_xlabel("일시", fontsize=11)
    ax.set_ylabel("발전량 (MWh)", fontsize=11)
    ax.set_ylim(-0.5, rated_capacity_mwh + 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved forecast plot to {save_path}")
    return fig, ax


def plot_horizon_metrics(
    horizons: list[int | str],
    metrics_by_model: dict[str, list[float]],
    metric_name: str = "NRMSE (%)",
    save_path: str | None = None,
):
    """
    Horizon 증가에 따른 각 모델별 성능 변화 비교 곡선을 생성합니다.
    """
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, values in metrics_by_model.items():
        ax.plot(horizons, values, marker="o", label=model_name, linewidth=1.8)

    ax.set_title(f"예측 Horizon에 따른 모델별 {metric_name} 변화", fontsize=13, pad=10)
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig, ax
