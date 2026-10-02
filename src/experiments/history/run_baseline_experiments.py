"""
상명풍력 발전량 M1 실험군(과거 발전량 기반) Baseline 및 LightGBM 예측 평가 스크립트
- Horizon: 1h, 3h, 6h, 12h, 24h
- 데이터 분할:
  - Train: 2023-01-01 ~ 2024-06-30
  - Validation: 2024-07-01 ~ 2024-12-31
  - Test: 2025-01-01 ~ 2025-12-31 (단위 보정 완료된 2025년 전체)
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.features.lag_features import create_horizon_target
from src.utils.metrics import evaluate_wind_forecast


def run_m1_benchmarks():
    data_path = "data/interim/generation_hourly.parquet"
    if not os.path.exists(data_path):
        from src.data.load_generation import build_and_save_generation_interim
        build_and_save_generation_interim(output_parquet=data_path)

    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 시계열 분할
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    horizons = [1, 3, 6, 12, 24]
    results = []

    # 1. Persistence Baseline
    naive = NaivePersistence()
    diurnal = DiurnalPersistence(period=24)

    test_series = df_test["generation_mwh"].copy()

    for h in horizons:
        y_true = create_horizon_target(df_test, target_col="generation_mwh", horizon=h)

        # Naive Persistence (y_hat_{t+h} = y_t)
        y_pred_naive = df_test["generation_mwh"]
        mask_n = y_true.notna() & y_pred_naive.notna()
        m_naive = evaluate_wind_forecast(y_true[mask_n], y_pred_naive[mask_n], rated_capacity_mwh=21.0)
        m_naive["model"] = "Naive_Persistence"
        m_naive["horizon"] = f"+{h}h"
        results.append(m_naive)

        # 24h Diurnal Persistence (y_hat_{t+h} = y_{t+h-24})
        y_pred_diurnal = diurnal.predict(df_test["generation_mwh"], horizon=h)
        mask_d = y_true.notna() & y_pred_diurnal.notna()
        m_diurnal = evaluate_wind_forecast(y_true[mask_d], y_pred_diurnal[mask_d], rated_capacity_mwh=21.0)
        m_diurnal["model"] = "24h_Diurnal_Persistence"
        m_diurnal["horizon"] = f"+{h}h"
        results.append(m_diurnal)

    # 2. LightGBM Forecaster
    print("Training LightGBM forecaster...")
    forecaster = SangmyeongLightGBMForecaster(horizons=horizons)
    
    # Feature engineering
    df_feat = forecaster.prepare_features(df)
    train_feat = df_feat[train_mask].copy().reset_index(drop=True)
    val_feat = df_feat[val_mask].copy().reset_index(drop=True)
    test_feat = df_feat[test_mask].copy().reset_index(drop=True)

    lgb_preds = {}
    for h in horizons:
        forecaster.train_horizon(train_feat, val_feat, horizon=h)
        m_lgb, y_eval, y_pred = forecaster.evaluate(test_feat, horizon=h)
        m_lgb["model"] = "LightGBM"
        m_lgb["horizon"] = f"+{h}h"
        results.append(m_lgb)
        lgb_preds[h] = (y_eval, y_pred)

    df_results = pd.DataFrame(results)
    col_order = [
        "model",
        "horizon",
        "mae",
        "rmse",
        "mse",
        "nmae_pct",
        "nrmse_pct",
        "rse",
        "rrse",
        "rae",
        "wape_pct",
        "smape_pct",
        "r2",
        "corr",
        "mbe",
        "tic",
        "count",
    ]
    df_results = df_results[[c for c in col_order if c in df_results.columns]]

    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    df_results.to_csv("reports/tables/baseline_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== Model Benchmark Results (Test 2025) ===")
    print(df_results.to_string(index=False))

    # 시각화: Horizon별 MAE 및 RMSE 비교
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, grp in df_results.groupby("model"):
        axes[0].plot(grp["horizon"], grp["mae"], marker="o", lw=2, label=model_name)
        axes[1].plot(grp["horizon"], grp["rmse"], marker="s", lw=2, label=model_name)

    axes[0].set_title("MAE by Forecast Horizon (MWh, Test 2025)", fontweight="bold")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("MAE (MWh)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("RMSE by Forecast Horizon (MWh, Test 2025)", fontweight="bold")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("RMSE (MWh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/figures/baseline_horizon_comparison.png", dpi=200)
    plt.close()

    # 1시간 예측 실제값 vs 예측값 1주 샘플 시각화
    y_true_1h, y_pred_1h = lgb_preds[1]
    plt.figure(figsize=(14, 5))
    sample_len = 168  # 1주일
    plt.plot(range(sample_len), y_true_1h[:sample_len], label="Actual (MWh)", color="black", lw=1.5)
    plt.plot(range(sample_len), y_pred_1h[:sample_len], label="LightGBM +1h Forecast", color="#1f77b4", lw=1.5, linestyle="--")
    plt.title("Sangmyeong Wind Generation: 1-Week Sample Forecast (+1h Ahead)", fontweight="bold")
    plt.xlabel("Hours")
    plt.ylabel("Generation (MWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/lightgbm_1h_forecast_sample.png", dpi=200)
    plt.close()

    print("\nBenchmark plots successfully generated in reports/figures/")
    return df_results


if __name__ == "__main__":
    run_m1_benchmarks()
