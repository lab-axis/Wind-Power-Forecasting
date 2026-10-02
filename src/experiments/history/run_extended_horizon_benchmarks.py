"""
Extended Horizon Benchmark Experiment Script: +12h and +24h Ahead
- Evaluates the multi-horizon forecasting spectrum:
  Ultra-short (+1h), Short (+3h), Medium (+12h), Day-Ahead (+24h)
- Models evaluated:
  1. Proposed: EMFN v3 (+Weather) [Canonical Mean, Bayes Median, Zero AUROC/AUPRC, Physical Bounds]
  2. Machine Learning: LightGBM (+Weather)
  3. Deep Learning: TCN (+Weather), LSTM (+Weather), DLinear (+Weather),
                    CNN-LSTM (+Weather), Transformer (+Weather)
  4. Physics/Stat: Naive Persistence, 24h Diurnal Persistence
- Protocol:
  - 2025 Test Set (8,760 hours, 100% complete, 0.00% missing)
- Outputs:
  - reports/tables/extended_horizon_12h_24h_benchmark_comparison.csv
  - reports/tables/full_multi_horizon_benchmark_matrix.csv
  - reports/figures/multi_horizon_performance_decay_curve.png
  - reports/figures/emfn_v3_forecast_horizon_12h_24h_sample.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.emfn_trainer import train_and_evaluate_emfn as train_and_evaluate_emfn_v3
from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.models.dlinear import DLinearForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.rnn import LSTMForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster
from src.models.torch_trainer import train_and_evaluate_torch_model
from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.features.lag_features import create_lag_features, create_rolling_features, create_horizon_target
from src.features.time_features import add_cyclical_time_features
from src.utils.metrics import evaluate_wind_forecast

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def main():
    print("=" * 85)
    print("   Extended Horizon Wind Power Benchmarks (+12h & +24h Ahead, 2025 Test Set)   ")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Data
    data_path = "data/processed/merged_dataset.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Master merged dataset not found at {data_path}.")

    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    weather_cols = [
        "aws_wind_speed",
        "aws_wind_dir_sin",
        "aws_wind_dir_cos",
        "aws_temperature",
        "aws_humidity",
        "aws_local_pressure",
    ]
    target_col = "generation_mwh"
    all_cols = [target_col] + weather_cols
    capacity_mwh = 21.0

    # 2. Strict Dataset Split Protocol (2025 8,760h Test Set Only, 2026 Excluded)
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f"[*] Splits: Train={len(df_train):,}h, Val={len(df_val):,}h, Test={len(df_test):,}h")

    # 3. Standardize Weather based on Train statistics
    w_mean = df_train[weather_cols].mean()
    w_std = df_train[weather_cols].std().replace(0, 1.0)

    for col in weather_cols:
        df_train[col] = (df_train[col] - w_mean[col]) / w_std[col]
        df_val[col] = (df_val[col] - w_mean[col]) / w_std[col]
        df_test[col] = (df_test[col] - w_mean[col]) / w_std[col]

    train_data = df_train[all_cols].values
    val_data = df_val[all_cols].values
    test_data = df_test[all_cols].values

    input_dim = len(all_cols)
    lookback = 24
    ext_horizons = [12, 24]

    results_extended = []

    # =========================================================================
    # A. Statistical Baselines: Naive & Diurnal Persistence (+12h, +24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 1. Statistical Baselines (Naive & 24h Diurnal Persistence)")
    print("=" * 80)
    naive = NaivePersistence()
    diurnal = DiurnalPersistence(period=24)

    for h in ext_horizons:
        y_true = create_horizon_target(df_test, target_col="generation_mwh", horizon=h)

        # Naive Persistence
        y_pred_naive = df_test["generation_mwh"]
        mask_n = y_true.notna() & y_pred_naive.notna()
        m_naive = evaluate_wind_forecast(y_true[mask_n], y_pred_naive[mask_n], rated_capacity_mwh=capacity_mwh)
        m_naive["model"] = "Naive Persistence"
        m_naive["horizon"] = f"+{h}h"
        m_naive["weather_used"] = False
        m_naive["bound_violation_pct"] = float(np.mean((y_pred_naive[mask_n] < 0.0) | (y_pred_naive[mask_n] > capacity_mwh)) * 100.0)
        results_extended.append(m_naive)
        print(f"  [+] Naive Persistence +{h}h -> MAE: {m_naive['mae']:.4f} MWh | RMSE: {m_naive['rmse']:.4f} MWh | R^2: {m_naive['r2']:.4f}")

        # 24h Diurnal Persistence
        y_pred_diurnal = diurnal.predict(df_test["generation_mwh"], horizon=h)
        mask_d = y_true.notna() & y_pred_diurnal.notna()
        m_diurnal = evaluate_wind_forecast(y_true[mask_d], y_pred_diurnal[mask_d], rated_capacity_mwh=capacity_mwh)
        m_diurnal["model"] = "24h Diurnal Persistence"
        m_diurnal["horizon"] = f"+{h}h"
        m_diurnal["weather_used"] = False
        m_diurnal["bound_violation_pct"] = float(np.mean((y_pred_diurnal[mask_d] < 0.0) | (y_pred_diurnal[mask_d] > capacity_mwh)) * 100.0)
        results_extended.append(m_diurnal)
        print(f"  [+] 24h Diurnal Persistence +{h}h -> MAE: {m_diurnal['mae']:.4f} MWh | RMSE: {m_diurnal['rmse']:.4f} MWh | R^2: {m_diurnal['r2']:.4f}")

    # =========================================================================
    # B. Machine Learning: LightGBM (+Weather Lags & Stats) (+12h, +24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 2. Training LightGBM with Weather Lags (+12h, +24h)")
    print("=" * 80)
    df_feat = df.copy()
    df_feat = add_cyclical_time_features(df_feat, time_col="datetime")
    df_feat = create_lag_features(df_feat, target_cols=["generation_mwh"], lags=[1, 2, 3, 6, 12, 24, 48, 168])
    df_feat = create_rolling_features(df_feat, target_cols=["generation_mwh"], windows=[3, 6, 24])
    df_feat = create_lag_features(df_feat, target_cols=["aws_wind_speed", "aws_temperature", "aws_local_pressure"], lags=[1, 2, 3, 6, 24])
    df_feat = create_rolling_features(df_feat, target_cols=["aws_wind_speed"], windows=[3, 6, 24])

    exclude_cols = {
        "datetime", "base_date", "hour", "raw_generation", "scale_unit_assumed",
        "is_capacity_exceeded", "is_negative", "is_zero"
    }

    train_lgb = df_feat[train_mask].copy().reset_index(drop=True)
    val_lgb = df_feat[val_mask].copy().reset_index(drop=True)
    test_lgb = df_feat[test_mask].copy().reset_index(drop=True)

    lgb_forecaster = SangmyeongLightGBMForecaster(horizons=ext_horizons)
    lgb_forecaster.feature_cols = [c for c in df_feat.columns if c not in exclude_cols and not c.startswith("target_h")]

    for h in ext_horizons:
        t0 = time.time()
        lgb_forecaster.train_horizon(train_lgb, val_lgb, horizon=h)
        m_lgb, y_t_lgb, y_p_lgb = lgb_forecaster.evaluate(test_lgb, horizon=h)
        elapsed = time.time() - t0
        m_lgb["model"] = "LightGBM (+Weather)"
        m_lgb["horizon"] = f"+{h}h"
        m_lgb["weather_used"] = True
        m_lgb["bound_violation_pct"] = float(np.mean((y_p_lgb < 0.0) | (y_p_lgb > capacity_mwh)) * 100.0)
        m_lgb["min_pred_mwh"] = float(np.min(y_p_lgb))
        m_lgb["train_time_sec"] = elapsed
        results_extended.append(m_lgb)
        print(f"  [+] LightGBM (+Weather) +{h}h ({elapsed:.1f}s) -> MAE: {m_lgb['mae']:.4f} MWh ({m_lgb['nmae_pct']:.2f}%) | "
              f"RMSE: {m_lgb['rmse']:.4f} MWh | R^2: {m_lgb['r2']:.4f} | Viol: {m_lgb['bound_violation_pct']:.2f}% (min {m_lgb['min_pred_mwh']:.3f})")

    # =========================================================================
    # C. Deep Learning Baselines with Weather (+12h, +24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 3. Training Deep Learning Models (+Weather) (+12h, +24h)")
    print("=" * 80)

    dl_configs = [
        ("DLinear (+Weather)", lambda: DLinearForecaster(lookback_steps=24, forecast_horizon=1, input_dim=input_dim)),
        ("CNN-LSTM (+Weather)", lambda: CNNLSTMForecaster(input_dim=input_dim, lookback_steps=24, forecast_horizon=1)),
        ("LSTM (+Weather)", lambda: LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2, forecast_horizon=1)),
        ("TCN (+Weather)", lambda: TCNForecaster(input_dim=input_dim, num_channels=[64, 64, 64], kernel_size=3, forecast_horizon=1)),
        ("Transformer (+Weather)", lambda: TimeSeriesTransformerForecaster(input_dim=input_dim, lookback_steps=24, forecast_horizon=1)),
    ]

    for model_name, model_fn in dl_configs:
        print(f"\n[*] Training {model_name}...")
        for h in ext_horizons:
            t0 = time.time()
            model = model_fn()
            m_dl, y_t_dl, y_p_dl = train_and_evaluate_torch_model(
                model=model,
                train_series=train_data,
                val_series=val_data,
                test_series=test_data,
                lookback_steps=24,
                horizon=h,
                epochs=25,
                batch_size=128,
                lr=0.002,
                rated_capacity_mwh=capacity_mwh,
                device=device,
            )
            elapsed = time.time() - t0
            m_dl["model"] = model_name
            m_dl["horizon"] = f"+{h}h"
            m_dl["weather_used"] = True
            m_dl["bound_violation_pct"] = float(np.mean((y_p_dl < 0.0) | (y_p_dl > capacity_mwh)) * 100.0)
            m_dl["min_pred_mwh"] = float(np.min(y_p_dl))
            m_dl["train_time_sec"] = elapsed
            results_extended.append(m_dl)
            print(f"  [+] {model_name} +{h}h ({elapsed:.1f}s) -> MAE: {m_dl['mae']:.4f} MWh ({m_dl['nmae_pct']:.2f}%) | "
                  f"RMSE: {m_dl['rmse']:.4f} MWh | R^2: {m_dl['r2']:.4f} | Viol: {m_dl['bound_violation_pct']:.2f}% (min {m_dl['min_pred_mwh']:.3f})")

    # =========================================================================
    # D. Proposed Model: EMFN v3 (+Weather) (+12h, +24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 4. Training Proposed EMFN v3 (+Weather) (+12h, +24h)")
    print("=" * 80)

    emfn_preds_dict = {}

    for h in ext_horizons:
        ckpt_path = f"models/checkpoints/emfn_v3_{h}h.pt"
        preds_ckpt = f"models/checkpoints/emfn_v3_{h}h_test_preds.pt"

        summary_v3, y_mean_v3, preds_v3 = train_and_evaluate_emfn_v3(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            lookback_steps=lookback,
            horizon=h,
            include_weather=True,
            epochs=35,
            patience=7,
            capacity_mwh=capacity_mwh,
            device=device,
            model_name="EMFN_v3 (+Weather)",
            save_model_path=ckpt_path,
        )
        torch.save(preds_v3, preds_ckpt)
        emfn_preds_dict[h] = preds_v3

        # Formatted row for comparison
        row_v3 = {
            "model": "EMFN v3 (Proposed)",
            "horizon": f"+{h}h",
            "weather_used": True,
            "mae": summary_v3["mae_canonical"],
            "rmse": summary_v3["rmse_canonical"],
            "nmae_pct": summary_v3["nmae_canonical_pct"],
            "r2": summary_v3["r2_canonical"],
            "corr": summary_v3["corr_canonical"],
            "wape_pct": summary_v3["wape_canonical_pct"],
            "mae_median": summary_v3["mae_median"],
            "median_gain_pct": summary_v3["median_gain_pct"],
            "zero_auroc": summary_v3["zero_auroc"],
            "zero_auprc": summary_v3["zero_auprc"],
            "zero_brier": summary_v3["zero_brier"],
            "zero_ece": summary_v3["zero_ece"],
            "bound_violation_pct": summary_v3["bound_violation_pct"],
            "min_pred_mwh": summary_v3["min_pred_mwh"],
            "train_time_sec": summary_v3["train_time_sec"],
        }
        results_extended.append(row_v3)
        print(f"  [+] EMFN v3 (Proposed) +{h}h -> Canonical MAE: {row_v3['mae']:.4f} MWh ({row_v3['nmae_pct']:.2f}%) | "
              f"RMSE: {row_v3['rmse']:.4f} MWh | R^2: {row_v3['r2']:.4f} | "
              f"Zero AUROC: {row_v3['zero_auroc']:.4f} | Zero AUPRC: {row_v3['zero_auprc']:.4f} | "
              f"Viol: {row_v3['bound_violation_pct']:.2f}% (min {row_v3['min_pred_mwh']:.3f}) | Median MAE: {row_v3['mae_median']:.4f}")

    # =========================================================================
    # E. Save Extended Results (+12h, +24h)
    # =========================================================================
    df_ext = pd.DataFrame(results_extended)
    ext_csv_path = "reports/tables/extended_horizon_12h_24h_benchmark_comparison.csv"
    os.makedirs(os.path.dirname(ext_csv_path), exist_ok=True)
    df_ext.to_csv(ext_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved +12h/+24h Extended Horizon Table to: {ext_csv_path}")

    # =========================================================================
    # F. Construct Full 4-Horizon Unified Benchmark Matrix (1h, 3h, 12h, 24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 5. Compiling Unified 4-Horizon Benchmark Matrix (+1h, +3h, +12h, +24h)")
    print("=" * 80)

    # Load previously verified results for +1h and +3h
    p_w_aug = "reports/tables/weather_augmented_benchmark_comparison.csv"
    p_v3_bm = "reports/tables/emfn_v3_benchmark_results.csv"

    rows_1h_3h = []
    if os.path.exists(p_w_aug):
        df_w_prev = pd.read_csv(p_w_aug)
        for _, r in df_w_prev.iterrows():
            rows_1h_3h.append(r.to_dict())

    # Add 1h, 3h EMFN v3
    if os.path.exists(p_v3_bm):
        df_v3_prev = pd.read_csv(p_v3_bm)
        for _, r in df_v3_prev.iterrows():
            if "Weather" in str(r.get("model", "")):
                d_row = r.to_dict()
                d_row["model"] = "EMFN v3 (Proposed)"
                d_row["mae"] = d_row.get("mae_canonical")
                d_row["rmse"] = d_row.get("rmse_canonical")
                d_row["nmae_pct"] = d_row.get("nmae_canonical_pct")
                d_row["r2"] = d_row.get("r2_canonical")
                d_row["corr"] = d_row.get("corr_canonical")
                d_row["wape_pct"] = d_row.get("wape_canonical_pct")
                rows_1h_3h.append(d_row)

    # Add Naive and Diurnal for 1h and 3h from baseline_model_comparison
    p_base = "reports/tables/baseline_model_comparison.csv"
    if os.path.exists(p_base):
        df_b_prev = pd.read_csv(p_base)
        for _, r in df_b_prev.iterrows():
            if str(r.get("model")) in ["Naive_Persistence", "24h_Diurnal_Persistence"] and str(r.get("horizon")) in ["+1h", "+3h"]:
                d_row = r.to_dict()
                d_row["model"] = "Naive Persistence" if "Naive" in d_row["model"] else "24h Diurnal Persistence"
                rows_1h_3h.append(d_row)

    df_full_matrix = pd.concat([pd.DataFrame(rows_1h_3h), df_ext], ignore_index=True)

    # Standardize column naming
    full_csv_path = "reports/tables/full_multi_horizon_benchmark_matrix.csv"
    df_full_matrix.to_csv(full_csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Saved Complete Multi-Horizon Matrix to: {full_csv_path}")

    # =========================================================================
    # G. Multi-Horizon Performance Decay Curves & Visualizations
    # =========================================================================
    plot_multi_horizon_decay(df_full_matrix, out_fig="reports/figures/multi_horizon_performance_decay_curve.png")
    plot_extended_time_series(df_test, emfn_preds_dict, out_fig="reports/figures/emfn_v3_forecast_horizon_12h_24h_sample.png")

    print("\n=== Extended Horizon Benchmarks Successfully Completed! ===")


def plot_multi_horizon_decay(df: pd.DataFrame, out_fig: str):
    """Plots MAE and R^2 decay curves across [1h, 3h, 12h, 24h] for key models."""
    plt.rc("font", family="Malgun Gothic" if os.name == "nt" else "DejaVu Sans")
    plt.rc("axes", unicode_minus=False)

    horizons_num = [1, 3, 12, 24]
    horizons_str = ["+1h", "+3h", "+12h", "+24h"]

    target_models = [
        ("EMFN (Proposed)", "#2563eb", "o", 2.2, "-"),
        ("LightGBM (+Weather)", "#059669", "s", 1.8, "--"),
        ("TCN (+Weather)", "#dc2626", "^", 1.8, "--"),
        ("LSTM (+Weather)", "#d97706", "d", 1.5, ":"),
        ("DLinear (+Weather)", "#7c3aed", "v", 1.5, ":"),
        ("Naive Persistence", "#64748b", "x", 1.4, "-."),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for m_name, color, marker, lw, ls in target_models:
        mae_vals, r2_vals = [], []
        sub_df = df[df["model"].str.contains(m_name.split()[0], case=False, na=False)]

        for h_str in horizons_str:
            row = sub_df[sub_df["horizon"] == h_str]
            if len(row) > 0:
                mae_vals.append(row["mae"].values[0])
                r2_vals.append(row["r2"].values[0])
            else:
                mae_vals.append(np.nan)
                r2_vals.append(np.nan)

        if not np.all(np.isnan(mae_vals)):
            ax1.plot(horizons_num, mae_vals, label=m_name, color=color, marker=marker, lw=lw, ls=ls, markersize=6)
        if not np.all(np.isnan(r2_vals)):
            ax2.plot(horizons_num, r2_vals, label=m_name, color=color, marker=marker, lw=lw, ls=ls, markersize=6)

    # MAE Panel
    ax1.set_title("Forecast Error Decay: MAE vs Horizon (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MAE (MWh)", fontsize=11, fontweight="bold")
    ax1.set_xticks(horizons_num)
    ax1.set_xticklabels(["+1h", "+3h", "+12h", "+24h"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # R^2 Panel
    ax2.set_title("Forecast Accuracy Retention: R² vs Horizon (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=11, fontweight="bold")
    ax2.set_xticks(horizons_num)
    ax2.set_xticklabels(["+1h", "+3h", "+12h", "+24h"])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    plt.suptitle("Jeju Sangmyeong Wind Power Multi-Horizon Forecasting Spectrum (2025 Test: 8,760h)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Multi-Horizon Decay Figure to: {out_fig}")


def plot_extended_time_series(df_test: pd.DataFrame, preds_dict: dict, out_fig: str):
    """Plots time-series forecast tracking for +12h and +24h."""
    fig, (ax12, ax24) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    zoom_slice = slice(240, 240 + 168)  # 1 week in January 2025
    t = df_test["datetime"].iloc[zoom_slice]

    # +12h Panel
    p12 = preds_dict.get(12)
    if p12:
        y_true_12 = p12["y_true"][zoom_slice]
        y_mean_12 = p12["y_mean"][zoom_slice]
        ax12.plot(t, y_true_12, label="Actual Generation (MWh)", color="#111827", lw=1.3)
        ax12.plot(t, y_mean_12, label="EMFN Canonical Mean (+12h Ahead)", color="#2563eb", lw=1.5, ls="--")
        ax12.axhline(21.0, color="#dc2626", ls=":", lw=1.0, label="Rated Capacity (21 MW)")
        ax12.set_ylabel("Power (MWh)", fontsize=10, fontweight="bold")
        ax12.set_title("Medium-Term Dispatch Forecast: +12h Ahead (1-Week Sample)", fontsize=11, fontweight="bold")
        ax12.grid(True, linestyle="--", alpha=0.45)
        ax12.legend(loc="upper left", frameon=True, fontsize=9)

    # +24h Panel
    p24 = preds_dict.get(24)
    if p24:
        y_true_24 = p24["y_true"][zoom_slice]
        y_mean_24 = p24["y_mean"][zoom_slice]
        ax24.plot(t, y_true_24, label="Actual Generation (MWh)", color="#111827", lw=1.3)
        ax24.plot(t, y_mean_24, label="EMFN Canonical Mean (+24h Ahead Day-Ahead)", color="#059669", lw=1.5, ls="--")
        ax24.axhline(21.0, color="#dc2626", ls=":", lw=1.0, label="Rated Capacity (21 MW)")
        ax24.set_ylabel("Power (MWh)", fontsize=10, fontweight="bold")
        ax24.set_title("Day-Ahead Market Clearing Forecast: +24h Ahead (1-Week Sample)", fontsize=11, fontweight="bold")
        ax24.grid(True, linestyle="--", alpha=0.45)
        ax24.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved 12h/24h Time Series Figure to: {out_fig}")


if __name__ == "__main__":
    main()
