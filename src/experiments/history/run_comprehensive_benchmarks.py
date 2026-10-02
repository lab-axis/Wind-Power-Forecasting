"""
SCI Q1급 시계열 및 풍력 발전량 예측 모델 10대 종합 벤치마크 실험 스크립트

포함 모델 (총 10종):
[고전 통계 및 베이스라인]
1. Naive Persistence
2. 24h Diurnal Persistence
3. ARIMA(24, 0, 0) (Box & Jenkins 1970)
4. Prophet (Taylor & Letham 2018 / Meta)

[머신러닝 GBDT]
5. LightGBM (Ke et al. 2017)

[딥러닝 시계열 모델군]
6. DLinear (Zeng et al. AAAI 2023)
7. TCN (Temporal Convolutional Network, Bai et al. 2018)
8. CNN-LSTM (Park et al. 2022)
9. LSTM (Hochreiter & Schmidhuber 1997 / Kim et al. 2021)
10. Time Series Transformer (Vaswani et al. 2017 / Informer 2021)
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.models.lstm import LSTMForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster
from src.models.dlinear import DLinearForecaster
from src.models.torch_trainer import train_and_evaluate_torch_model
from src.features.lag_features import create_horizon_target
from src.utils.metrics import evaluate_wind_forecast


def run_all_benchmarks():
    data_path = "data/interim/generation_hourly.parquet"
    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 시계열 분할 (2025년 온전한 1년 Test 기준, Holdout 배제)
    train_mask = (df["datetime"] >= "2023-01-01 01:00:00") & (df["datetime"] <= "2024-06-30 23:00:00")
    val_mask = (df["datetime"] >= "2024-07-01 00:00:00") & (df["datetime"] <= "2024-12-31 23:00:00")
    test_mask = (df["datetime"] >= "2025-01-01 00:00:00") & (df["datetime"] <= "2025-12-31 23:00:00")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    train_series = df_train["generation_mwh"].values
    val_series = df_val["generation_mwh"].values
    test_series = df_test["generation_mwh"].values

    test_start_idx = df.index[test_mask][0]
    test_end_idx = df.index[test_mask][-1] + 1

    horizons = [1, 3, 6, 24]
    results = []

    print("=" * 80)
    print(" 1. Statistical Baselines (Naive & Diurnal Persistence)")
    print("=" * 80)
    naive = NaivePersistence()
    diurnal = DiurnalPersistence(period=24)

    for h in horizons:
        y_true = create_horizon_target(df_test, target_col="generation_mwh", horizon=h)

        # Naive Persistence
        y_pred_naive = df_test["generation_mwh"]
        mask_n = y_true.notna() & y_pred_naive.notna()
        m_naive = evaluate_wind_forecast(y_true[mask_n], y_pred_naive[mask_n], rated_capacity_mwh=21.0)
        m_naive["model"] = "Naive_Persistence"
        m_naive["horizon"] = f"+{h}h"
        results.append(m_naive)

        # 24h Diurnal Persistence
        y_pred_diurnal = diurnal.predict(df_test["generation_mwh"], horizon=h)
        mask_d = y_true.notna() & y_pred_diurnal.notna()
        m_diurnal = evaluate_wind_forecast(y_true[mask_d], y_pred_diurnal[mask_d], rated_capacity_mwh=21.0)
        m_diurnal["model"] = "24h_Diurnal_Persistence"
        m_diurnal["horizon"] = f"+{h}h"
        results.append(m_diurnal)

    print("=" * 80)
    print(" 2. Classical Econometric Baseline (ARIMA)")
    print("=" * 80)
    try:
        arima = ARIMAForecaster(p_lags=24, rated_capacity_mwh=21.0)
        arima.fit(df_train["generation_mwh"])
        for h in [1, 3]:
            y_pred_arima = arima.predict_horizon(df["generation_mwh"], test_start_idx, test_end_idx, horizon=h)
            y_true_arima = create_horizon_target(df_test, target_col="generation_mwh", horizon=h).values
            m_arima = evaluate_wind_forecast(y_true_arima[:len(y_pred_arima)], y_pred_arima, rated_capacity_mwh=21.0)
            m_arima["model"] = "ARIMA(24,0,0)"
            m_arima["horizon"] = f"+{h}h"
            results.append(m_arima)
    except Exception as e:
        print(f"ARIMA run error: {e}")

    print("=" * 80)
    print(" 3. Additive Decomposition Model (Prophet)")
    print("=" * 80)
    try:
        prophet_model = ProphetForecaster(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            rated_capacity_mwh=21.0,
        )
        prophet_model.fit(df_train, time_col="datetime", target_col="generation_mwh")
        y_pred_prophet = prophet_model.predict(df_test, time_col="datetime")
        y_true_prophet = df_test["generation_mwh"].values
        m_prophet = evaluate_wind_forecast(y_true_prophet, y_pred_prophet, rated_capacity_mwh=21.0)
        m_prophet["model"] = "Prophet (Meta)"
        m_prophet["horizon"] = "+1h"
        results.append(m_prophet)
    except Exception as e:
        print(f"Prophet run error: {e}")

    print("=" * 80)
    print(" 4. Tabular Gradient Boosting (LightGBM)")
    print("=" * 80)
    forecaster = SangmyeongLightGBMForecaster(horizons=horizons)
    df_feat = forecaster.prepare_features(df)
    train_feat = df_feat[train_mask].copy().reset_index(drop=True)
    val_feat = df_feat[val_mask].copy().reset_index(drop=True)
    test_feat = df_feat[test_mask].copy().reset_index(drop=True)

    for h in horizons:
        forecaster.train_horizon(train_feat, val_feat, horizon=h)
        m_lgb, _, _ = forecaster.evaluate(test_feat, horizon=h)
        m_lgb["model"] = "LightGBM"
        m_lgb["horizon"] = f"+{h}h"
        results.append(m_lgb)

    print("=" * 80)
    print(" 5. Deep Learning Sequence Models (DLinear, TCN, LSTM, CNN-LSTM, Transformer)")
    print("=" * 80)
    dl_models = [
        ("DLinear (AAAI'23)", lambda: DLinearForecaster(lookback_steps=24, forecast_horizon=1, input_dim=1)),
        ("TCN (Bai'18)", lambda: TCNForecaster(input_dim=1, num_channels=[64, 64, 64], kernel_size=3, forecast_horizon=1)),
        ("LSTM (Hochreiter'97)", lambda: LSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, forecast_horizon=1)),
        ("CNN-LSTM (Park'22)", lambda: CNNLSTMForecaster(input_dim=1, lookback_steps=24, forecast_horizon=1)),
        ("Transformer (Vaswani'17)", lambda: TimeSeriesTransformerForecaster(input_dim=1, lookback_steps=24, forecast_horizon=1)),
    ]

    for model_name, model_fn in dl_models:
        print(f"Training {model_name}...")
        for h in [1, 3]:  # 핵심 Horizon 평가
            model = model_fn()
            m_dl, _, _ = train_and_evaluate_torch_model(
                model=model,
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                lookback_steps=24,
                horizon=h,
                epochs=20,
                batch_size=128,
                lr=0.002,
                rated_capacity_mwh=21.0,
            )
            m_dl["model"] = model_name
            m_dl["horizon"] = f"+{h}h"
            results.append(m_dl)

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
    df_results.sort_values(by=["horizon", "mae"], inplace=True)

    os.makedirs("reports/tables", exist_ok=True)
    out_csv = "reports/tables/comprehensive_model_benchmark.csv"
    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_results.to_csv("reports/tables/baseline_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(" 10-MODEL COMPREHENSIVE BENCHMARK EVALUATION (Test 2025: 8,760 Hours) ")
    print("=" * 100)
    print(df_results.to_string(index=False))

    return df_results


if __name__ == "__main__":
    run_all_benchmarks()
