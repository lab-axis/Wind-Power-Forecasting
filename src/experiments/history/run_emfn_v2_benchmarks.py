"""
EMFN v2 Benchmark Experiment Script
- Compares:
  1. EMFN v2 (+Weather): Multi-channel TCN + Weather Gating + Hurdle Beta Head
  2. EMFN v2 (Endogenous Only): TCN Target Only + Hurdle Beta Head (Ablation)
- Evaluates on 2025 Test Set (8,760 hours) across +1h and +3h
- Compares directly against Weather-TCN (1.4579 MWh) and Weather-LightGBM (1.4720 MWh)
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.history.emfn_v2_trainer import train_and_evaluate_emfn_v2

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def main():
    print("=" * 80)
    print("   EMFN v2 (Multi-channel TCN + Weather Gating + Hurdle Beta) Benchmark   ")
    print("=" * 80)

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

    # 2. Splits
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f"[*] Train set: {len(df_train)} hours ({df_train['datetime'].min()} ~ {df_train['datetime'].max()})")
    print(f"[*] Val set:   {len(df_val)} hours ({df_val['datetime'].min()} ~ {df_val['datetime'].max()})")
    print(f"[*] Test set:  {len(df_test)} hours ({df_test['datetime'].min()} ~ {df_test['datetime'].max()})")

    # 3. Standardize Weather based on Train Set
    w_mean = df_train[weather_cols].mean()
    w_std = df_train[weather_cols].std().replace(0, 1.0)

    for col in weather_cols:
        df_train[col] = (df_train[col] - w_mean[col]) / w_std[col]
        df_val[col] = (df_val[col] - w_mean[col]) / w_std[col]
        df_test[col] = (df_test[col] - w_mean[col]) / w_std[col]

    train_data = df_train[all_cols].values
    val_data = df_val[all_cols].values
    test_data = df_test[all_cols].values

    horizons = [1, 3]
    experiments = [
        ("EMFN_v2 (+Weather)", True),
        ("EMFN_v2 (Endogenous Only)", False),
    ]

    all_results = []
    saved_preds = {}

    for name, incl_weather in experiments:
        print(f"\n=======================================================")
        print(f"[*] Running Experiment: {name} (include_weather={incl_weather})")
        print(f"=======================================================")

        for h in horizons:
            print(f"  --> Training for Horizon +{h}h Ahead...")
            summary, y_mean, preds_dict = train_and_evaluate_emfn_v2(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                lookback_steps=24,
                horizon=h,
                include_weather=incl_weather,
                d_model=64,
                dilations=(1, 2, 4, 8),
                lambda_point=1.0,
                lr=0.001,
                batch_size=128,
                epochs=35,
                patience=6,
                capacity_mwh=21.0,
                device=device,
            )
            all_results.append(summary)
            saved_preds[f"{name}_h{h}"] = preds_dict

            print(f"    [+] Canonical Point: MAE={summary['mae_canonical']:.4f} MWh ({summary['nmae_canonical_pct']:.2f}%) | "
                  f"RMSE={summary['rmse_canonical']:.4f} MWh | R^2={summary['r2_canonical']:.4f}")
            print(f"    [+] Bayes Median:    MAE={summary['mae_median']:.4f} MWh (Gain: {summary['median_gain_pct']:.2f}%)")
            print(f"    [+] Hurdle Zero:     AUROC={summary['zero_auroc']:.4f} | AUPRC={summary['zero_auprc']:.4f} | "
                  f"ECE={summary['zero_ece']:.4f} | Bounds Violation={summary['bound_violation_pct']:.2f}%")

    # 4. Save Results Table
    df_res = pd.DataFrame(all_results)
    out_table_path = "reports/tables/emfn_v2_benchmark_results.csv"
    os.makedirs(os.path.dirname(out_table_path), exist_ok=True)
    df_res.to_csv(out_table_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Results saved to {out_table_path}")

    # 5. Visualization: Forecast Samples & Hurdle Probability
    print("\n[*] Generating visualization plot...")
    plt.figure(figsize=(14, 8))

    sample_slice = slice(200, 368)  # 1-week sample in test set
    timestamps = df_test["datetime"].iloc[sample_slice]

    p_w_h1 = saved_preds["EMFN_v2 (+Weather)_h1"]
    p_w_h3 = saved_preds["EMFN_v2 (+Weather)_h3"]

    plt.subplot(2, 1, 1)
    plt.plot(timestamps, p_w_h1["y_true"][sample_slice], label="Observed Generation", color="black", lw=1.8)
    plt.plot(timestamps, p_w_h1["y_mean"][sample_slice], label="EMFN v2 (+1h Mean)", color="#1f77b4", lw=1.5, alpha=0.9)
    plt.plot(timestamps, p_w_h3["y_mean"][sample_slice], label="EMFN v2 (+3h Mean)", color="#ff7f0e", lw=1.5, ls="--", alpha=0.9)
    plt.ylabel("Generation (MWh)", fontsize=11, fontweight="bold")
    plt.title("EMFN v2: Point Forecasts (0 <= Y <= 21 MWh Bounded)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.ylim(-0.5, 21.5)

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, p_w_h1["p_zero"][sample_slice], label="Predicted Zero Probability P(Y=0)", color="#d62728", lw=1.5)
    plt.fill_between(timestamps, 0, p_w_h1["p_zero"][sample_slice], color="#d62728", alpha=0.2)
    is_zero_obs = (p_w_h1["y_true"][sample_slice] < 1e-4).astype(float)
    plt.scatter(timestamps[is_zero_obs == 1], np.ones(np.sum(is_zero_obs == 1)) * 0.95, color="darkred", marker="x", s=30, label="Observed Zero State")
    plt.ylabel("P(Y = 0 | X)", fontsize=11, fontweight="bold")
    plt.xlabel("Datetime", fontsize=11, fontweight="bold")
    plt.title("EMFN v2: Zero-Generation Hurdle State Probability", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    out_fig_path = "reports/figures/emfn_v2_forecast_hurdle_sample.png"
    plt.savefig(out_fig_path, dpi=300)
    plt.close()
    print(f"[+] Figure saved to {out_fig_path}")


if __name__ == "__main__":
    main()
