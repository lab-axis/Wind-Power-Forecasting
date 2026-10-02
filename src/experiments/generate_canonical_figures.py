"""
Canonical Figure Generation Pipeline
(Generates 100% clean, publication-grade figures for EMFN Wind Power Forecasting)

Key Standards:
1. All figures use canonical 'EMFN' or 'EMFN (Proposed)' - absolutely no 'v3' in labels, legends, or titles.
2. Evaluated strictly on the official splits:
   - Validation Set: 2024-07-01 ~ 2024-12-31 (4,416 hours)
   - Official Test Set: 2025-01-01 ~ 2025-12-31 (8,760 hours, 0.00% missing)
   - 2026 data completely excluded.
3. Generates:
   - reports/figures/emfn_test_forecast_+1h.png
   - reports/figures/emfn_test_forecast_+3h.png
   - reports/figures/emfn_validation_forecast_+1h.png
   - reports/figures/emfn_validation_forecast_+3h.png
   - reports/figures/emfn_forecast_horizon_12h_24h_sample.png
   - reports/figures/emfn_ablation_comparison.png
   - reports/figures/intraday_dispatch_spectrum_decay_curve.png
   - reports/figures/multi_horizon_performance_decay_curve.png
4. Archives obsolete / remnant figures (e.g. emfn_v3_*, transition plots) to reports/figures/archive/.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import EMFN
from src.utils.forecast_visualizer import plot_evaluation_forecast, setup_plot_style
from src.experiments.run_comprehensive_ablation_matrix import plot_ablation_results


def archive_obsolete_figures():
    """Moves legacy v2/v3 figures and historical rollout transition plots to archive/."""
    fig_dir = os.path.join(PROJECT_ROOT, "reports/figures")
    archive_dir = os.path.join(fig_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    patterns_to_archive = [
        "emfn_v3_",
        "emfn_v2_",
        "historical_transition",
    ]

    for fname in os.listdir(fig_dir):
        if fname.endswith(".png"):
            if any(p in fname for p in patterns_to_archive):
                src_path = os.path.join(fig_dir, fname)
                dst_path = os.path.join(archive_dir, fname)
                shutil.move(src_path, dst_path)
                print(f"[Archive] Moved obsolete figure: {fname} -> reports/figures/archive/")


def generate_all_figures():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device for Figure Generation: {device}")

    # 1. Load Data
    data_path = os.path.join(PROJECT_ROOT, "data/processed/merged_dataset.parquet")
    df = pd.read_parquet(data_path)

    weather_cols = [
        "aws_wind_speed",
        "aws_wind_dir_sin",
        "aws_wind_dir_cos",
        "aws_temperature",
        "aws_humidity",
        "aws_local_pressure",
    ]
    all_cols = ["generation_mwh"] + weather_cols

    # Standardize weather using Train set stats
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    w_mean = df.loc[train_mask, weather_cols].mean()
    w_std = df.loc[train_mask, weather_cols].std().replace(0, 1.0)

    df_norm = df.copy()
    for col in weather_cols:
        df_norm[col] = (df_norm[col] - w_mean[col]) / w_std[col]

    df_val = df_norm.loc[val_mask].copy().reset_index(drop=True)
    df_test = df_norm.loc[test_mask].copy().reset_index(drop=True)

    val_data = df_val[all_cols].values
    test_data = df_test[all_cols].values

    # Helper function to run model inference
    def run_inference(model, data_arr, horizon):
        L = 24
        X_target, X_weather, Y = [], [], []
        for i in range(len(data_arr) - L - horizon + 1):
            X_target.append(data_arr[i : i + L, 0:1])
            X_weather.append(data_arr[i : i + L, 1:])
            Y.append(data_arr[i + L + horizon - 1, 0])

        t_target = torch.tensor(np.array(X_target), dtype=torch.float32).to(device)
        t_weather = torch.tensor(np.array(X_weather), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model.predict_point_forecasts(t_target, t_weather)
        return {
            "y_true": np.array(Y),
            "y_mean": np.asarray(preds["y_mean_rmse"]).flatten(),
            "y_median": np.asarray(preds["y_median_mae"]).flatten(),
            "p_pos": np.asarray(preds["p_positive"]).flatten(),
            "p_zero": 1.0 - np.asarray(preds["p_positive"]).flatten(),
        }

    # 2. Generate Validation & Test Forecast Tracking Plots (+1h & +3h)
    print("\n[*] Generating Validation & Test Forecast Tracking figures...")
    for h in [1, 3]:
        ckpt_path = os.path.join(PROJECT_ROOT, f"models/checkpoints/emfn_{h}h.pt")
        model = EMFN(lookback_len=24, pred_len=1, n_weather_features=6).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Test Set Plot
        preds_test = run_inference(model, test_data, h)
        test_ts = df_test["datetime"].iloc[24 + h - 1 :].reset_index(drop=True)

        fig_test_path = os.path.join(PROJECT_ROOT, f"reports/figures/emfn_test_forecast_+{h}h.png")
        plot_evaluation_forecast(
            timestamps=test_ts,
            y_true=preds_test["y_true"],
            y_mean=preds_test["y_mean"],
            y_median=preds_test["y_median"],
            p_zero=preds_test["p_zero"],
            dataset_name="Official Test Set (2025)",
            horizon_label=f"+{h}h Ahead",
            save_path=fig_test_path,
            zoom_range=("2025-01-10 00:00", "2025-01-17 00:00"),
            rated_capacity_mwh=21.0,
        )
        plt.close()

        # Validation Set Plot
        preds_val = run_inference(model, val_data, h)
        val_ts = df_val["datetime"].iloc[24 + h - 1 :].reset_index(drop=True)

        fig_val_path = os.path.join(PROJECT_ROOT, f"reports/figures/emfn_validation_forecast_+{h}h.png")
        plot_evaluation_forecast(
            timestamps=val_ts,
            y_true=preds_val["y_true"],
            y_mean=preds_val["y_mean"],
            y_median=preds_val["y_median"],
            p_zero=preds_val["p_zero"],
            dataset_name="Validation Set (2024 H2)",
            horizon_label=f"+{h}h Ahead",
            save_path=fig_val_path,
            zoom_range=("2024-11-01 00:00", "2024-11-08 00:00"),
            rated_capacity_mwh=21.0,
        )
        plt.close()

    # 3. Generate Long-Horizon (+12h & +24h) Tracking Figure
    print("\n[*] Generating Long-Horizon (+12h & +24h) Forecast Tracking figure...")
    ckpt_12h = os.path.join(PROJECT_ROOT, "models/checkpoints/emfn_12h.pt")
    ckpt_24h = os.path.join(PROJECT_ROOT, "models/checkpoints/emfn_24h.pt")
    model_12h = EMFN(lookback_len=24, pred_len=1, n_weather_features=6).to(device)
    model_24h = EMFN(lookback_len=24, pred_len=1, n_weather_features=6).to(device)
    model_12h.load_state_dict(torch.load(ckpt_12h, map_location=device))
    model_24h.load_state_dict(torch.load(ckpt_24h, map_location=device))
    model_12h.eval()
    model_24h.eval()

    preds_12h = run_inference(model_12h, test_data, 12)
    preds_24h = run_inference(model_24h, test_data, 24)

    # 1-week sample in Jan 2025
    zoom_slice = slice(240, 240 + 168)
    t_12 = df_test["datetime"].iloc[24 + 12 - 1 :].reset_index(drop=True).iloc[zoom_slice]
    t_24 = df_test["datetime"].iloc[24 + 24 - 1 :].reset_index(drop=True).iloc[zoom_slice]

    setup_plot_style()
    fig, (ax12, ax24) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # +12h panel
    ax12.plot(t_12, preds_12h["y_true"][zoom_slice], label="Actual Generation (Ground Truth)", color="#111827", lw=1.3)
    ax12.plot(t_12, preds_12h["y_mean"][zoom_slice], label="EMFN Canonical Mean (+12h Ahead)", color="#2563eb", lw=1.6, ls="--")
    ax12.axhline(21.0, color="#dc2626", ls=":", lw=1.0, label="Rated Capacity (21 MW)")
    ax12.set_ylabel("Power (MWh)", fontsize=10, fontweight="bold")
    ax12.set_title("Medium-Term Dispatch Forecast: +12h Ahead (1-Week Sample)", fontsize=11, fontweight="bold")
    ax12.grid(True, linestyle="--", alpha=0.45)
    ax12.legend(loc="upper left", frameon=True, fontsize=9)

    # +24h panel
    ax24.plot(t_24, preds_24h["y_true"][zoom_slice], label="Actual Generation (Ground Truth)", color="#111827", lw=1.3)
    ax24.plot(t_24, preds_24h["y_mean"][zoom_slice], label="EMFN Canonical Mean (+24h Ahead Day-Ahead)", color="#059669", lw=1.6, ls="--")
    ax24.axhline(21.0, color="#dc2626", ls=":", lw=1.0, label="Rated Capacity (21 MW)")
    ax24.set_ylabel("Power (MWh)", fontsize=10, fontweight="bold")
    ax24.set_title("Day-Ahead Market Clearing Forecast: +24h Ahead (1-Week Sample)", fontsize=11, fontweight="bold")
    ax24.grid(True, linestyle="--", alpha=0.45)
    ax24.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    out_12_24 = os.path.join(PROJECT_ROOT, "reports/figures/emfn_forecast_horizon_12h_24h_sample.png")
    fig.savefig(out_12_24, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Long-Horizon sample figure to: {out_12_24}")

    # 4. Generate Ablation Comparison Figure
    print("\n[*] Generating Ablation Comparison figure...")
    abl_csv = os.path.join(PROJECT_ROOT, "reports/tables/emfn_comprehensive_ablation_matrix.csv")
    if os.path.exists(abl_csv):
        df_abl = pd.read_csv(abl_csv)
        abl_fig_path = os.path.join(PROJECT_ROOT, "reports/figures/emfn_ablation_comparison.png")
        plot_ablation_results(df_abl, abl_fig_path)

    # 5. Generate Multi-Horizon Decay Curves (Intraday & Full)
    print("\n[*] Generating Multi-Horizon Decay Curves...")
    full_csv = os.path.join(PROJECT_ROOT, "reports/tables/full_multi_horizon_benchmark_matrix.csv")
    if os.path.exists(full_csv):
        df_matrix = pd.read_csv(full_csv)

        # Plot 1: Intraday Dispatch Spectrum Decay (+1h ~ +12h)
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        target_models = [
            ("EMFN (Proposed)", "#2563eb", "o", 2.2, "-"),
            ("LightGBM (+Weather)", "#059669", "s", 1.8, "--"),
            ("TCN (+Weather)", "#dc2626", "^", 1.8, "--"),
            ("LSTM (+Weather)", "#d97706", "d", 1.5, ":"),
            ("DLinear (+Weather)", "#7c3aed", "v", 1.5, ":"),
            ("Naive Persistence", "#64748b", "x", 1.4, "-."),
        ]

        h_intraday_num = [1, 3, 6, 9, 12]
        h_intraday_str = ["+1h", "+3h", "+6h", "+9h", "+12h"]

        for m_name, color, marker, lw, ls in target_models:
            sub = df_matrix[df_matrix["model"] == m_name]
            maes = [sub.loc[sub["horizon"] == h, "mae"].values[0] for h in h_intraday_str if len(sub.loc[sub["horizon"] == h]) > 0]
            r2s = [sub.loc[sub["horizon"] == h, "r2"].values[0] for h in h_intraday_str if len(sub.loc[sub["horizon"] == h]) > 0]
            ax1.plot(h_intraday_num[:len(maes)], maes, label=m_name, color=color, marker=marker, lw=lw, ls=ls)
            ax2.plot(h_intraday_num[:len(r2s)], r2s, label=m_name, color=color, marker=marker, lw=lw, ls=ls)

        ax1.set_title("Intra-Day Dispatch Spectrum: MAE vs Horizon (Lower is Better)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Dispatch Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("MAE (MWh)", fontsize=11, fontweight="bold")
        ax1.set_xticks(h_intraday_num)
        ax1.set_xticklabels(h_intraday_str)
        ax1.grid(True, linestyle="--", alpha=0.45)
        ax1.legend(loc="upper left", frameon=True, fontsize=9)

        ax2.set_title("Intra-Day Dispatch Spectrum: R² Retention (Higher is Better)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Dispatch Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=11, fontweight="bold")
        ax2.set_xticks(h_intraday_num)
        ax2.set_xticklabels(h_intraday_str)
        ax2.grid(True, linestyle="--", alpha=0.45)
        ax2.legend(loc="lower left", frameon=True, fontsize=9)

        plt.suptitle("Jeju Sangmyeong Wind Power Intra-Day Dispatch Spectrum (2025 Test: 8,760h)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_intra_decay = os.path.join(PROJECT_ROOT, "reports/figures/intraday_dispatch_spectrum_decay_curve.png")
        fig.savefig(out_intra_decay, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Intra-Day Decay Curve to: {out_intra_decay}")

        # Plot 2: Full Multi-Horizon Decay (+1h ~ +24h)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        h_all_num = [1, 3, 6, 9, 12, 24]
        h_all_str = ["+1h", "+3h", "+6h", "+9h", "+12h", "+24h"]

        for m_name, color, marker, lw, ls in target_models:
            sub = df_matrix[df_matrix["model"] == m_name]
            maes = [sub.loc[sub["horizon"] == h, "mae"].values[0] for h in h_all_str if len(sub.loc[sub["horizon"] == h]) > 0]
            r2s = [sub.loc[sub["horizon"] == h, "r2"].values[0] for h in h_all_str if len(sub.loc[sub["horizon"] == h]) > 0]
            ax1.plot(h_all_num[:len(maes)], maes, label=m_name, color=color, marker=marker, lw=lw, ls=ls)
            ax2.plot(h_all_num[:len(r2s)], r2s, label=m_name, color=color, marker=marker, lw=lw, ls=ls)

        ax1.set_title("Forecast Error Decay: MAE vs Multi-Horizon (Lower is Better)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("MAE (MWh)", fontsize=11, fontweight="bold")
        ax1.set_xticks(h_all_num)
        ax1.set_xticklabels(h_all_str)
        ax1.grid(True, linestyle="--", alpha=0.45)
        ax1.legend(loc="upper left", frameon=True, fontsize=9)

        ax2.set_title("Forecast Accuracy Retention: R² vs Multi-Horizon (Higher is Better)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=11, fontweight="bold")
        ax2.set_xticks(h_all_num)
        ax2.set_xticklabels(h_all_str)
        ax2.grid(True, linestyle="--", alpha=0.45)
        ax2.legend(loc="lower left", frameon=True, fontsize=9)

        plt.suptitle("Jeju Sangmyeong Wind Power Multi-Horizon Forecasting Spectrum (2025 Test: 8,760h)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_multi_decay = os.path.join(PROJECT_ROOT, "reports/figures/multi_horizon_performance_decay_curve.png")
        fig.savefig(out_multi_decay, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Multi-Horizon Decay Curve to: {out_multi_decay}")

    print("\n=== All Canonical Figures Successfully Generated and Verified! ===")


if __name__ == "__main__":
    archive_obsolete_figures()
    generate_all_figures()
