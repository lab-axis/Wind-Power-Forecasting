"""
Comprehensive 4-Axis Ablation Matrix Experiment Script for EMFN
- Evaluates on 2025 Test Set (8,760 hours, 100% complete, 0% missing)
- Horizons: +1h Ahead and +3h Ahead
- Ablation Axes:
  1. Full EMFN (+Weather, Proposed) [Proposed Baseline]
  2. EMFN (Endogenous Only) [Axis 1: Meteorological Conditioning]
  3. EMFN (w/o HF Skips) [Axis 2: High-Frequency Momentum Skips & AR Shortcut]
  4. EMFN (w/o Selective Gate) [Axis 3: Domain-Specific Weather Gating for Zero Head]
  5. EMFN (Deterministic Huber Regressor) [Axis 4: Physical Hurdle Head vs Unconstrained Regression]
- Outputs:
  - reports/tables/emfn_comprehensive_ablation_matrix.csv
  - reports/figures/emfn_ablation_comparison.png
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

from src.models.emfn_trainer import train_and_evaluate_emfn

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def main():
    print("=" * 85)
    print("   EMFN Comprehensive 4-Axis Ablation Matrix Experiment (2025 Test Set)   ")
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

    # 2. Strict Dataset Split Protocol (2025 8,760h Test Set Only, 2026 Excluded)
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f"[*] Dataset Splits:")
    print(f"    - Train: {len(df_train):,} hours (2023-01-01 ~ 2024-06-30)")
    print(f"    - Val:   {len(df_val):,} hours (2024-07-01 ~ 2024-12-31)")
    print(f"    - Test:  {len(df_test):,} hours (2025-01-01 ~ 2025-12-31, Missing: 0.00%)")

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

    # Ablation configurations
    configs = [
        {
            "name": "Full EMFN (+Weather, Proposed)",
            "include_weather": True,
            "use_hf_skips": True,
            "use_selective_gate": True,
            "regression_mode": False,
            "description": "Full Task-Decoupled Dual-Route Hurdle Network",
        },
        {
            "name": "EMFN (Endogenous Only)",
            "include_weather": False,
            "use_hf_skips": True,
            "use_selective_gate": True,
            "regression_mode": False,
            "description": "Ablation: Exclude AWS meteorological features",
        },
        {
            "name": "EMFN (w/o HF Skips)",
            "include_weather": True,
            "use_hf_skips": False,
            "use_selective_gate": True,
            "regression_mode": False,
            "description": "Ablation: Exclude high-frequency momentum skips & AR shortcut",
        },
        {
            "name": "EMFN (w/o Selective Gate)",
            "include_weather": True,
            "use_hf_skips": True,
            "use_selective_gate": False,
            "regression_mode": False,
            "description": "Ablation: Exclude domain-specific cut-in gating for zero head",
        },
        {
            "name": "EMFN (Deterministic Regression)",
            "include_weather": True,
            "use_hf_skips": True,
            "use_selective_gate": True,
            "regression_mode": True,
            "description": "Ablation: Unconstrained Linear + Huber loss (No Hurdle)",
        },
    ]

    all_results = []
    saved_preds = {}

    for cfg in configs:
        c_name = cfg["name"]
        incl_w = cfg["include_weather"]
        hf_skips = cfg["use_hf_skips"]
        sel_gate = cfg["use_selective_gate"]
        reg_mode = cfg["regression_mode"]

        print(f"\n=======================================================")
        print(f"[*] Running Configuration: {c_name}")
        print(f"    - Weather: {incl_w} | HF Skips: {hf_skips} | Selective Gate: {sel_gate} | Reg Mode: {reg_mode}")
        print(f"=======================================================")

        for h in horizons:
            print(f"  --> Training for Horizon +{h}h Ahead...")
            summary, y_mean, preds_dict = train_and_evaluate_emfn(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                lookback_steps=24,
                horizon=h,
                include_weather=incl_w,
                d_model=64,
                dilations=(1, 2, 4, 8),
                lambda_point=2.0,
                lr=1e-3,
                batch_size=128,
                epochs=35,
                patience=7,
                capacity_mwh=21.0,
                device=device,
                use_hf_skips=hf_skips,
                use_selective_gate=sel_gate,
                regression_mode=reg_mode,
                model_name=c_name,
            )

            print(
                f"      [+{h}h] Canonical MAE: {summary['mae_canonical']:.4f} MWh ({summary['nmae_canonical_pct']:.2f}%) | "
                f"RMSE: {summary['rmse_canonical']:.4f} MWh | R^2: {summary['r2_canonical']:.4f} | "
                f"Zero AUROC: {summary['zero_auroc'] if not np.isnan(summary['zero_auroc']) else 'N/A'} | "
                f"Zero AUPRC: {summary['zero_auprc'] if not np.isnan(summary['zero_auprc']) else 'N/A'} | "
                f"Bound Viol: {summary['bound_violation_pct']:.2f}% (min {summary['min_pred_mwh']:.3f})"
            )

            all_results.append(summary)
            saved_preds[(c_name, h)] = preds_dict

    # Save to DataFrame
    df_results = pd.DataFrame(all_results)
    out_table_path = "reports/tables/emfn_comprehensive_ablation_matrix.csv"
    os.makedirs(os.path.dirname(out_table_path), exist_ok=True)
    df_results.to_csv(out_table_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved Comprehensive Ablation Matrix Table to: {out_table_path}")

    # Generate Comparative Visualization
    plot_ablation_results(df_results, out_fig_path="reports/figures/emfn_ablation_comparison.png")

    print("\n=== Comprehensive Ablation Matrix Run Completed Successfully! ===")


def plot_ablation_results(df_res: pd.DataFrame, out_fig_path: str):
    """
    Plots MAE, R^2, Zero AUPRC, and Physical Violations across ablation models.
    """
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_1h = df_res[df_res["horizon"] == "+1h"].copy()
    df_3h = df_res[df_res["horizon"] == "+3h"].copy()

    model_names_short = [
        "Full EMFN",
        "w/o Weather",
        "w/o HF Skips",
        "w/o Gate",
        "Regression",
    ]

    x = np.arange(len(model_names_short))
    width = 0.35

    # 1. MAE Comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, df_1h["mae_canonical"], width, label="+1h Ahead", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width/2, df_3h["mae_canonical"], width, label="+3h Ahead", color="#ff7f0e", alpha=0.85)
    ax1.set_title("Canonical MAE (MWh, Lower is Better)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend()

    # 2. R^2 Comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, df_1h["r2_canonical"], width, label="+1h Ahead", color="#2ca02c", alpha=0.85)
    ax2.bar(x + width/2, df_3h["r2_canonical"], width, label="+3h Ahead", color="#d62728", alpha=0.85)
    ax2.set_title("Coefficient of Determination R^2 (Higher is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend()

    # 3. Zero AUPRC (Detection Precision)
    ax3 = axes[1, 0]
    # Replace NaN with 0 for regression
    auprc_1h = df_1h["zero_auprc"].fillna(0.0)
    auprc_3h = df_3h["zero_auprc"].fillna(0.0)
    ax3.bar(x - width/2, auprc_1h, width, label="+1h Ahead", color="#9467bd", alpha=0.85)
    ax3.bar(x + width/2, auprc_3h, width, label="+3h Ahead", color="#8c564b", alpha=0.85)
    ax3.axhline(0.175, color="red", linestyle=":", label="Random Baseline (17.5%)")
    ax3.set_title("Zero AUPRC (Zero-State Precision, Higher is Better)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.legend()

    # 4. Bound Violation Rate (%)
    ax4 = axes[1, 1]
    viol_1h = df_1h["bound_violation_pct"]
    viol_3h = df_3h["bound_violation_pct"]
    ax4.bar(x - width/2, viol_1h, width, label="+1h Ahead", color="#e377c2", alpha=0.85)
    ax4.bar(x + width/2, viol_3h, width, label="+3h Ahead", color="#7f7f7f", alpha=0.85)
    ax4.set_title("Physical Bound Violation Rate (%, Strictly 0% Target)", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax4.legend()

    plt.suptitle("EMFN Comprehensive 4-Axis Ablation Study (2025 Test: 8,760h)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Ablation Comparison Figure to: {out_fig_path}")


if __name__ == "__main__":
    main()
