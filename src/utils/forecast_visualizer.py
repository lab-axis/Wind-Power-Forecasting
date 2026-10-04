"""
Publication-Quality Forecast Visualization Module for Wind Power Generation
- Plots Validation and Test evaluation curves with Hurdle Zero probabilities.
- Plots seamless transition from Historical context into future Forecast Horizons.
- Designed for top-tier academic journals (IEEE Transactions, Applied Energy).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def setup_plot_style():
    """Configures high-DPI matplotlib styling with Korean and English font support."""
    try:
        plt.rc("font", family="Malgun Gothic")
    except Exception:
        plt.rc("font", family="DejaVu Sans")
    plt.rc("axes", unicode_minus=False)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10


def plot_evaluation_forecast(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_median: Optional[np.ndarray] = None,
    p_zero: Optional[np.ndarray] = None,
    dataset_name: str = "Test Set (2025)",
    horizon_label: str = "+1h Ahead",
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    zoom_range: Optional[Tuple[str, str]] = None,
    rated_capacity_mwh: float = 21.0,
) -> plt.Figure:
    """
    Generates a dual-panel evaluation figure:
    - Top panel: Actual vs Predicted Power Generation (Canonical Mean & Bayes Median)
                 with physical capacity bounds and error/performance metric box.
    - Bottom panel: Aligned Hurdle Zero Probability P(Y = 0 | X) indicating operational cut-in/cut-out.
    """
    setup_plot_style()

    df_plot = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps),
        "y_true": y_true,
        "y_mean": y_mean,
    })
    if y_median is not None:
        df_plot["y_median"] = y_median
    if p_zero is not None:
        df_plot["p_zero"] = p_zero

    # Optional slice zooming
    if zoom_range is not None:
        start_dt, end_dt = pd.to_datetime(zoom_range[0]), pd.to_datetime(zoom_range[1])
        mask = (df_plot["datetime"] >= start_dt) & (df_plot["datetime"] <= end_dt)
        df_plot = df_plot[mask].copy().reset_index(drop=True)

    fig, (ax_main, ax_zero) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.08}
    )

    t = df_plot["datetime"]

    # 1. Top Panel: Generation Tracking
    ax_main.plot(t, df_plot["y_true"], label="Actual Generation (Ground Truth)", color="#111827", lw=1.3, alpha=0.9)
    ax_main.plot(t, df_plot["y_mean"], label=f"EMFN Canonical Mean E[Y|X] ({horizon_label})", color="#2563eb", lw=1.4, alpha=0.95)

    if "y_median" in df_plot.columns:
        ax_main.plot(t, df_plot["y_median"], label="EMFN Bayes Median (L1-optimal)", color="#10b981", lw=1.1, ls="--", alpha=0.85)

    # Physical Bounds
    ax_main.axhline(rated_capacity_mwh, color="#dc2626", ls=":", lw=1.2, label=f"Rated Capacity ({rated_capacity_mwh} MW)")
    ax_main.axhline(0.0, color="#6b7280", ls="-", lw=0.8, alpha=0.6)

    ax_main.set_ylim(-0.5, rated_capacity_mwh + 1.2)
    ax_main.set_ylabel("Power Generation (MWh)", fontsize=11, fontweight="bold")
    ax_main.grid(True, linestyle="--", alpha=0.45)
    ax_main.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.9, fontsize=9.5)

    # Title
    zoom_text = f" [Zoom: {zoom_range[0]} ~ {zoom_range[1]}]" if zoom_range else ""
    ax_main.set_title(f"EMFN Wind Power Forecast Tracking: {dataset_name} ({horizon_label}){zoom_text}",
                      fontsize=13, fontweight="bold", pad=10)

    # Metrics Box
    if metrics:
        mae_val = metrics.get("mae", np.mean(np.abs(df_plot["y_true"] - df_plot["y_mean"])))
        rmse_val = metrics.get("rmse", np.sqrt(np.mean((df_plot["y_true"] - df_plot["y_mean"]) ** 2)))
        r2_val = metrics.get("r2", 1.0 - np.sum((df_plot["y_true"] - df_plot["y_mean"])**2) / np.sum((df_plot["y_true"] - np.mean(df_plot["y_true"]))**2))
        viol_val = metrics.get("bound_violation_pct", 0.0)

        metric_box_str = (
            f"MAE: {mae_val:.4f} MWh\n"
            f"RMSE: {rmse_val:.4f} MWh\n"
            f"R²: {r2_val:.4f}\n"
            f"Bound Viol: {viol_val:.2f}%"
        )
        ax_main.text(
            0.985, 0.95, metric_box_str,
            transform=ax_main.transAxes,
            fontsize=9.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92)
        )

    # 2. Bottom Panel: Zero Probability
    if "p_zero" in df_plot.columns:
        ax_zero.fill_between(t, 0, df_plot["p_zero"], color="#f59e0b", alpha=0.45, label="P(Y = 0 | X) [Zero-State Probability]")
        ax_zero.plot(t, df_plot["p_zero"], color="#d97706", lw=1.0)
        ax_zero.axhline(0.5, color="#b45309", ls=":", lw=0.9, label="Decision Threshold (0.5)")
        ax_zero.set_ylabel("P(Zero)", fontsize=10, fontweight="bold")
        ax_zero.set_ylim(-0.05, 1.05)
        ax_zero.grid(True, linestyle="--", alpha=0.4)
        ax_zero.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=8.5)

    # Date formatting
    ax_zero.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:00"))
    ax_zero.set_xlabel("Time (Hourly)", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved evaluation forecast plot to: {save_path}")

    return fig


def plot_history_to_future_forecast(
    history_timestamps: pd.DatetimeIndex,
    history_generation: np.ndarray,
    future_timestamps: pd.DatetimeIndex,
    future_forecast_mean: np.ndarray,
    future_forecast_median: Optional[np.ndarray] = None,
    future_actual: Optional[np.ndarray] = None,
    future_p_zero: Optional[np.ndarray] = None,
    origin_timestamp: Optional[str] = None,
    horizon_label: str = "+1h to +24h Forecast Rollout",
    rated_capacity_mwh: float = 21.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots historical observations seamlessly transitioning into future forecast horizons:
    - Left region: Historical Ground Truth (Dark Charcoal / Solid line) up to origin T_0.
    - Vertical line: Forecast Origin (T_0) where history ends and future rollout begins.
    - Right region: Forecast Horizon into the future with Canonical Mean, Bayes Median, and optional Ground Truth.
    - Lower panel: Predicted Zero Probability across the forecast horizon.
    """
    setup_plot_style()

    hist_t = np.asarray(pd.to_datetime(history_timestamps))
    fut_t = np.asarray(pd.to_datetime(future_timestamps))
    history_generation = np.asarray(history_generation)
    future_forecast_mean = np.asarray(future_forecast_mean)
    future_forecast_median = np.asarray(future_forecast_median) if future_forecast_median is not None else None
    future_actual = np.asarray(future_actual) if future_actual is not None else None
    future_p_zero = np.asarray(future_p_zero) if future_p_zero is not None else None

    if origin_timestamp is None:
        origin_timestamp = str(hist_t[-1])
    origin_dt = pd.to_datetime(origin_timestamp)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.08}
    )

    # --- Background Region Shading ---
    # History shaded area
    ax_top.axvspan(hist_t[0], origin_dt, color="#f1f5f9", alpha=0.7, label="Observed Historical Context")
    # Future forecast shaded area
    ax_top.axvspan(origin_dt, fut_t[-1], color="#eff6ff", alpha=0.7, label="Future Forecast Window")

    ax_bot.axvspan(hist_t[0], origin_dt, color="#f1f5f9", alpha=0.7)
    ax_bot.axvspan(origin_dt, fut_t[-1], color="#eff6ff", alpha=0.7)

    # --- Historical Series ---
    ax_top.plot(hist_t, history_generation, label="Historical Actual (MWh)", color="#1e293b", lw=1.8, marker="o", markersize=2.5)

    # --- Seamless Joint Point at Origin T_0 ---
    last_hist_val = history_generation[-1]
    connect_t = [hist_t[-1], fut_t[0]]
    connect_mean = [last_hist_val, future_forecast_mean[0]]
    ax_top.plot(connect_t, connect_mean, color="#2563eb", ls="--", lw=1.5, alpha=0.8)

    # --- Future Forecast Series ---
    ax_top.plot(fut_t, future_forecast_mean, label=f"EMFN v3 Forecast Mean E[Y|X]", color="#2563eb", lw=2.0, ls="--", marker="s", markersize=3.5)

    if future_forecast_median is not None:
        ax_top.plot(fut_t, future_forecast_median, label="EMFN v3 Forecast Median", color="#10b981", lw=1.4, ls=":", marker="^", markersize=3.0)

    if future_actual is not None:
        ax_top.plot(fut_t, future_actual, label="Future Observed Ground Truth", color="#64748b", lw=1.2, ls="-", alpha=0.75)

    # --- Origin Divider Line ---
    ax_top.axvline(origin_dt, color="#dc2626", ls="-.", lw=1.8, label="Forecast Origin T_0 (Cut-off)")
    ax_bot.axvline(origin_dt, color="#dc2626", ls="-.", lw=1.8)

    # Physical bounds
    ax_top.axhline(rated_capacity_mwh, color="#9ca3af", ls=":", lw=1.2, label=f"Rated Capacity ({rated_capacity_mwh} MW)")
    ax_top.axhline(0.0, color="#cbd5e1", ls="-", lw=0.8)

    ax_top.set_ylim(-0.5, rated_capacity_mwh + 1.2)
    ax_top.set_ylabel("Power Generation (MWh)", fontsize=11, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.45)
    ax_top.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92, fontsize=9)
    ax_top.set_title(f"Seamless Transition: Historical Sequence to Future Forecast ({horizon_label})\n"
                     f"Anchor Origin: {origin_dt.strftime('%Y-%m-%d %H:%M')}", fontsize=12, fontweight="bold", pad=8)

    # --- Bottom Panel: Zero Probability across Forecast Horizon ---
    if future_p_zero is not None:
        # Zero history dummy (0 if generation > 0, 1 if == 0)
        hist_zero = (history_generation == 0.0).astype(float)
        ax_bot.plot(hist_t, hist_zero, color="#64748b", lw=1.2, label="Historical Zero State (Binary)")

        # Connect at origin
        conn_pzero = [hist_zero[-1], future_p_zero[0]]
        ax_bot.plot(connect_t, conn_pzero, color="#d97706", ls="--", lw=1.2)

        ax_bot.fill_between(fut_t, 0, future_p_zero, color="#f59e0b", alpha=0.45, label="Predicted P(Y = 0 | X)")
        ax_bot.plot(fut_t, future_p_zero, color="#d97706", lw=1.8, marker="o", markersize=3.0)
        ax_bot.axhline(0.5, color="#b45309", ls=":", lw=0.9, label="Cut-off (0.5)")

        ax_bot.set_ylabel("P(Zero)", fontsize=10, fontweight="bold")
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.grid(True, linestyle="--", alpha=0.4)
        ax_bot.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=8.5)

    # Formatting x-axis dates
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax_bot.set_xlabel("Timeline (Historical Context -> Future Forecast)", fontsize=11, fontweight="bold")

    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.07, right=0.98, hspace=0.10)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved historical transition forecast plot to: {save_path}")

    return fig
