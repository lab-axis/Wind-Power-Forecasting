"""
Weather-Augmented Baseline Benchmarks Comparison Experiment
- Injects real Saebyeol-Oreum AWS 883 weather features (wind speed, wind direction sin/cos,
  temperature, humidity, local pressure) into all key baselines:
  1. LightGBM (+ Weather Lags & Stats)
  2. DLinear (+ Multivariate Weather Input)
  3. CNN-LSTM (+ Multivariate Weather Input)
  4. LSTM (+ Multivariate Weather Input)
  5. TCN (+ Multivariate Weather Input)
  6. Transformer (+ Multivariate Weather Input)
  7. EMFN (Proposed Multi-Scale Cross-Attention + Hurdle Head)
- Evaluates on 2025 Test (8,760 hours) across horizons (+1h, +3h)
- Directly quantifies the delta: With Weather vs Without Weather!
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.models.dlinear import DLinearForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.rnn import LSTMForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster
from src.models.torch_trainer import train_and_evaluate_torch_model
from src.features.lag_features import create_lag_features, create_rolling_features, create_horizon_target
from src.features.time_features import add_cyclical_time_features
from src.utils.metrics import evaluate_wind_forecast


def run_weather_benchmarks():
    print("=" * 80)
    print("   Weather-Augmented (Saebyeol-Oreum AWS 883) Benchmark Evaluation    ")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Data
    data_path = "data/processed/merged_dataset.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Master merged dataset not found at {data_path}. Run process_saebyeol_weather.py first.")

    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Weather Columns
    weather_cols = [
        "aws_wind_speed",
        "aws_wind_dir_sin",
        "aws_wind_dir_cos",
        "aws_temperature",
        "aws_humidity",
        "aws_local_pressure",
    ]
    print(f"[*] Weather features ({len(weather_cols)}): {weather_cols}")

    # 3. Splits
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    # 4. Standardize weather based on Train statistics
    w_mean = df_train[weather_cols].mean()
    w_std = df_train[weather_cols].std().replace(0, 1.0)

    norm_w_train = ((df_train[weather_cols] - w_mean) / w_std).values
    norm_w_val = ((df_val[weather_cols] - w_mean) / w_std).values
    norm_w_test = ((df_test[weather_cols] - w_mean) / w_std).values

    # Multichannel matrices: Target (column 0) + Weather (columns 1..6)
    train_multi = np.column_stack([df_train["generation_mwh"].values, norm_w_train])
    val_multi = np.column_stack([df_val["generation_mwh"].values, norm_w_val])
    test_multi = np.column_stack([df_test["generation_mwh"].values, norm_w_test])

    input_dim = train_multi.shape[1]
    print(f"[*] Multichannel input dimension: {input_dim} (1 generation target + {len(weather_cols)} AWS weather)")

    horizons = [1, 3]
    results = []

    # =========================================================================
    # A. LightGBM with Weather Lags & Stats
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 1. Training LightGBM with Real AWS 883 Weather Lags")
    print("=" * 80)

    # Prepare features with generation AND weather lags
    df_feat = df.copy()
    df_feat = add_cyclical_time_features(df_feat, time_col="datetime")
    df_feat = create_lag_features(df_feat, target_cols=["generation_mwh"], lags=[1, 2, 3, 6, 12, 24, 48, 168])
    df_feat = create_rolling_features(df_feat, target_cols=["generation_mwh"], windows=[3, 6, 24])

    # Add weather lags for wind speed, temperature, pressure
    df_feat = create_lag_features(df_feat, target_cols=["aws_wind_speed", "aws_temperature", "aws_local_pressure"], lags=[1, 2, 3, 6, 24])
    df_feat = create_rolling_features(df_feat, target_cols=["aws_wind_speed"], windows=[3, 6, 24])

    exclude_cols = {
        "datetime", "base_date", "hour", "raw_generation", "scale_unit_assumed",
        "is_capacity_exceeded", "is_negative", "is_zero"
    }

    train_lgb_feat = df_feat[train_mask].copy().reset_index(drop=True)
    val_lgb_feat = df_feat[val_mask].copy().reset_index(drop=True)
    test_lgb_feat = df_feat[test_mask].copy().reset_index(drop=True)

    lgb_forecaster = SangmyeongLightGBMForecaster(horizons=horizons)
    lgb_forecaster.feature_cols = [c for c in df_feat.columns if c not in exclude_cols and not c.startswith("target_h")]
    print(f"[*] LightGBM features with weather ({len(lgb_forecaster.feature_cols)} features)")

    for h in horizons:
        lgb_forecaster.train_horizon(train_lgb_feat, val_lgb_feat, horizon=h)
        m_lgb, _, _ = lgb_forecaster.evaluate(test_lgb_feat, horizon=h)
        m_lgb["model"] = "LightGBM (+Weather)"
        m_lgb["horizon"] = f"+{h}h"
        m_lgb["weather_used"] = True
        results.append(m_lgb)
        print(f"  [+] LightGBM (+Weather) +{h}h -> MAE: {m_lgb['mae']:.4f} MWh ({m_lgb['nmae_pct']:.2f}%) | RMSE: {m_lgb['rmse']:.4f} MWh | R^2: {m_lgb['r2']:.4f}")

    # =========================================================================
    # B. Deep Learning Models with Multivariate Weather (input_dim = 7)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" 2. Training Deep Learning Models with Multivariate AWS Weather (input_dim=7)")
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
        for h in horizons:
            t0 = time.time()
            model = model_fn()
            m_dl, _, _ = train_and_evaluate_torch_model(
                model=model,
                train_series=train_multi,
                val_series=val_multi,
                test_series=test_multi,
                lookback_steps=24,
                horizon=h,
                epochs=25,
                batch_size=128,
                lr=0.002,
                rated_capacity_mwh=21.0,
                device=device,
            )
            elapsed = time.time() - t0
            m_dl["model"] = model_name
            m_dl["horizon"] = f"+{h}h"
            m_dl["weather_used"] = True
            m_dl["train_time_sec"] = elapsed
            results.append(m_dl)
            print(f"  [+] {model_name} +{h}h ({elapsed:.1f}s) -> MAE: {m_dl['mae']:.4f} MWh ({m_dl['nmae_pct']:.2f}%) | RMSE: {m_dl['rmse']:.4f} MWh | R^2: {m_dl['r2']:.4f}")

    # =========================================================================
    # C. Compile Comparison Table
    # =========================================================================
    df_aug = pd.DataFrame(results)

    # Load univariate baseline results for comparison
    base_csv = "reports/tables/comprehensive_model_benchmark.csv"
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
    else:
        df_base = pd.DataFrame()

    os.makedirs("reports/tables", exist_ok=True)
    aug_csv_path = "reports/tables/weather_augmented_benchmark_comparison.csv"
    df_aug.to_csv(aug_csv_path, index=False)
    print(f"\n[+] Saved weather-augmented results to: {aug_csv_path}")

    # Print summary table
    cols = ["model", "horizon", "mae", "nmae_pct", "rmse", "nrmse_pct", "r2", "corr", "wape_pct"]
    print("\n" + "=" * 80)
    print("             WEATHER-AUGMENTED BENCHMARK RESULTS (2025 TEST)            ")
    print("=" * 80)
    print(df_aug[[c for c in cols if c in df_aug.columns]].to_string(index=False))

    return df_aug


if __name__ == "__main__":
    run_weather_benchmarks()
