"""
Master Execution Script for EMFN (Bernoulli-Beta Hurdle) Training Pipeline
- Trains EMFN on Train set (2023-01 to 2024-06)
- Performs Early Stopping and Hyperparameter Selection on Validation set (2024-07 to 2024-12)
- Evaluates Final Frozen Model on Test set (2025-01 to 2025-12, 8,760 hours)
- Compares with top baselines (LightGBM, DLinear, CNN-LSTM)
- Evaluates point forecast metrics (MAE mixture median, RMSE mixture mean, R^2, nMAE)
  and Zero Head classification metrics (AUROC, AUPRC, F1, Brier)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.features.time_features import add_cyclical_time_features
from src.models.history.emfn_v1 import EMFN
from src.models.history.emfn_v1_trainer import train_emfn_model


def run_pipeline():
    print("=" * 80)
    print("   EMFN (Exogenous Multi-scale Fusion Network) Training Pipeline   ")
    print("        with Numerically Stable Bernoulli-Beta Hurdle Head         ")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Data (Master Merged Dataset with Saebyeol-Oreum 883 AWS Weather)
    data_path = "data/processed/merged_dataset.parquet"
    if not os.path.exists(data_path):
        data_path = "data/interim/generation_hourly.parquet"

    print(f"[*] Loading master dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Select Exogenous Covariates (Real AWS 883 Meteorological + Cyclical Time Features)
    weather_cols = [
        "aws_wind_speed",
        "aws_wind_dir_sin",
        "aws_wind_dir_cos",
        "aws_temperature",
        "aws_humidity",
        "aws_local_pressure",
    ]
    time_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]
    
    # Ensure time features are present
    if "hour_sin" not in df.columns:
        df = add_cyclical_time_features(df, time_col="datetime")

    exog_cols = [c for c in weather_cols + time_cols if c in df.columns]

    # 3. Splits
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    train_target = df_train["generation_mwh"].values
    val_target = df_val["generation_mwh"].values
    test_target = df_test["generation_mwh"].values

    # Normalize continuous exogenous variables strictly based on Train split statistics (no leakage)
    scaler_mean = df_train[exog_cols].mean()
    scaler_std = df_train[exog_cols].std().replace(0, 1.0)

    train_exog = ((df_train[exog_cols] - scaler_mean) / scaler_std).values
    val_exog = ((df_val[exog_cols] - scaler_mean) / scaler_std).values
    test_exog = ((df_test[exog_cols] - scaler_mean) / scaler_std).values

    print(f"[*] Train hours: {len(df_train):,} ({df_train['datetime'].min()} ~ {df_train['datetime'].max()})")
    print(f"[*] Val hours:   {len(df_val):,} ({df_val['datetime'].min()} ~ {df_val['datetime'].max()})")
    print(f"[*] Test hours:  {len(df_test):,} ({df_test['datetime'].min()} ~ {df_test['datetime'].max()})")
    print(f"[*] Exogenous features ({len(exog_cols)} total, including {len([c for c in weather_cols if c in exog_cols])} AWS weather): {exog_cols}")

    horizons = [1, 3]
    all_results = []
    trained_models = {}

    for h in horizons:
        print("\n" + "-" * 70)
        print(f"   Training EMFN (Hurdle Head) for Horizon +{h}h Ahead")
        print("-" * 70)

        # 1. Full EMFN (Endogenous + Exogenous Cross-Attention + Hurdle Head)
        t0 = time.time()
        model_full = EMFN(
            lookback_len=168,
            pred_len=1,
            n_exog_features=len(exog_cols),
            d_model=64,
            n_heads=4,
            dilations=(1, 2, 4, 8),
            capacity_mwh=21.0,
            dropout=0.1,
        )

        res_full = train_emfn_model(
            model=model_full,
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            train_exog=train_exog,
            val_exog=val_exog,
            test_exog=test_exog,
            lookback_len=168,
            horizon=h,
            epochs=35,
            batch_size=128,
            lr=1e-3,
            patience=8,
            device=device,
        )
        elapsed_full = time.time() - t0
        res_full["model_variant"] = "EMFN_Full (Exog CA + Hurdle)"
        res_full["train_time_sec"] = elapsed_full
        all_results.append(res_full)
        trained_models[f"EMFN_h{h}"] = (res_full, model_full)

        print(f"  [+] Finished EMFN Full (+{h}h) in {elapsed_full:.1f}s | Best Val NLL: {res_full['best_val_nll']:.4f}")
        print(f"      Canonical E[Y|X]: MAE {res_full['mae_canonical']:.4f} MWh ({res_full['nmae_canonical_pct']:.2f}%) | RMSE {res_full['rmse_canonical']:.4f} MWh ({res_full['nrmse_canonical_pct']:.2f}%) | R^2: {res_full['r2_canonical']:.4f}")
        print(f"      Bayes Median:     MAE {res_full['mae_median']:.4f} MWh ({res_full['nmae_median_pct']:.2f}%, gain {res_full['mae_median_improvement_pct']:.2f}%)")
        print(f"      Zero Head:        AUROC {res_full['zero_auroc']:.4f} | AUPRC {res_full['zero_auprc']:.4f} | Brier {res_full['zero_brier']:.4f} | ECE {res_full['zero_ece']:.4f}")

        # 2. Ablation: EMFN Endogenous Only (n_exog_features=0)
        t0 = time.time()
        model_endo = EMFN(
            lookback_len=168,
            pred_len=1,
            n_exog_features=0,
            d_model=64,
            n_heads=4,
            dilations=(1, 2, 4, 8),
            capacity_mwh=21.0,
            dropout=0.1,
        )

        res_endo = train_emfn_model(
            model=model_endo,
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            train_exog=None,
            val_exog=None,
            test_exog=None,
            lookback_len=168,
            horizon=h,
            epochs=35,
            batch_size=128,
            lr=1e-3,
            patience=8,
            device=device,
        )
        elapsed_endo = time.time() - t0
        res_endo["model_variant"] = "EMFN_Ablation (Endogenous Only + Hurdle)"
        res_endo["train_time_sec"] = elapsed_endo
        all_results.append(res_endo)

        print(f"  [+] Finished EMFN Endo (+{h}h) in {elapsed_endo:.1f}s | Best Val NLL: {res_endo['best_val_nll']:.4f}")
        print(f"      Canonical E[Y|X]: MAE {res_endo['mae_canonical']:.4f} MWh ({res_endo['nmae_canonical_pct']:.2f}%) | RMSE {res_endo['rmse_canonical']:.4f} MWh ({res_endo['nrmse_canonical_pct']:.2f}%) | R^2: {res_endo['r2_canonical']:.4f}")
        print(f"      Bayes Median:     MAE {res_endo['mae_median']:.4f} MWh ({res_endo['nmae_median_pct']:.2f}%, gain {res_endo['mae_median_improvement_pct']:.2f}%)")
        print(f"      Zero Head:        AUROC {res_endo['zero_auroc']:.4f} | AUPRC {res_endo['zero_auprc']:.4f} | Brier {res_endo['zero_brier']:.4f} | ECE {res_endo['zero_ece']:.4f}")

    # 4. Format & Save Results Table
    report_rows = []
    for r in all_results:
        report_rows.append({
            "Horizon": r["horizon"],
            "Model Variant": r["model_variant"],
            "MAE_Canonical (MWh)": round(r["mae_canonical"], 4),
            "nMAE_Canonical (%)": round(r["nmae_canonical_pct"], 2),
            "RMSE_Canonical (MWh)": round(r["rmse_canonical"], 4),
            "nRMSE_Canonical (%)": round(r["nrmse_canonical_pct"], 2),
            "R^2_Canonical": round(r["r2_canonical"], 4),
            "CORR": round(r["corr_canonical"], 4),
            "WAPE_Canonical (%)": round(r["wape_canonical_pct"], 2),
            "MAE_Median (MWh)": round(r["mae_median"], 4),
            "Median_Gain (%)": round(r["mae_median_improvement_pct"], 2),
            "Zero_AUROC": round(r["zero_auroc"], 4),
            "Zero_AUPRC": round(r["zero_auprc"], 4),
            "Zero_Brier": round(r["zero_brier"], 4),
            "Zero_ECE": round(r["zero_ece"], 4),
            "Epochs": r["epochs_trained"],
            "Train_Time (s)": round(r["train_time_sec"], 1),
        })

    df_report = pd.DataFrame(report_rows)
    os.makedirs("reports/tables", exist_ok=True)
    df_report.to_csv("reports/tables/emfn_benchmark_results.csv", index=False)
    print("\n" + "=" * 80)
    print("                     EMFN FINAL BENCHMARK RESULTS TABLE                    ")
    print("=" * 80)
    print(df_report.to_string(index=False))

    # 5. Visualization
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(15, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1st horizon snippet (first 168 hours of test = 1 week)
    res_h1 = trained_models["EMFN_h1"][0]
    sample_len = 168
    idx = np.arange(sample_len)

    plt.plot(idx, res_h1["y_true"][:sample_len], label="Ground Truth (2025 Test)", color="black", linewidth=2.0)
    plt.plot(idx, res_h1["y_pred_mean"][:sample_len], label="EMFN Mean (RMSE Optimal)", color="#1f77b4", linestyle="--", linewidth=1.8)
    plt.plot(idx, res_h1["y_pred_median"][:sample_len], label="EMFN Median (MAE Optimal)", color="#2ca02c", linestyle="-.", linewidth=1.8)

    # Fill Zero Prob
    ax2 = plt.gca().twinx()
    p_zero = 1.0 - res_h1["p_positive"][:sample_len]
    ax2.fill_between(idx, 0, p_zero, color="red", alpha=0.15, label="P(Y = 0) Zero Probability")
    ax2.set_ylabel("Predicted Zero Probability P(Y = 0)", color="red", fontsize=11)
    ax2.set_ylim(0, 1.05)

    plt.title("Sangmyeong Wind Farm (21 MW) - EMFN Hurdle Forecast & Zero Probability (+1h Ahead, 1-Week Test Snippet)", fontsize=13, fontweight="bold")
    plt.xlabel("Test Hour Index (First Week of 2025)", fontsize=11)
    plt.gca().set_ylabel("Wind Generation (MWh)", fontsize=11)
    plt.gca().set_ylim(0, 21.5)

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = "reports/figures/emfn_forecast_hurdle_sample.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"\n[+] Forecast visualization figure saved to: {fig_path}")

    return df_report


if __name__ == "__main__":
    run_pipeline()
