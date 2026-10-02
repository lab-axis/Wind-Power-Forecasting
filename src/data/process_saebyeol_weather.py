"""
Saebyeol-Oreum (AWS 883) Hourly Weather Data Processing and Dataset Merging Module
- Loads raw hourly CSVs (2023, 2024, 2025)
- Standardizes meteorological column names
- Computes circular wind direction encodings (sin/cos)
- Imputes minor missing values (< 0.2%) via time-based linear interpolation
- Merges with generation_hourly.parquet to generate data/processed/merged_dataset.parquet
"""

import os
import sys
import glob
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.features.time_features import add_cyclical_time_features
from src.features.wind_direction import wind_direction_to_sin_cos


def load_and_clean_saebyeol_weather(
    weather_dirs: Optional[list] = None,
) -> pd.DataFrame:
    """
    Loads, cleans, and standardizes Saebyeol-Oreum AWS 883 hourly observations.
    """
    if weather_dirs is None:
        weather_dirs = [
            "data/raw/weather/saebyeol_883-hour",
            "data/raw/weather/saebyeol_883",
        ]

    csv_files = []
    for d in weather_dirs:
        if os.path.exists(d):
            csv_files.extend(glob.glob(os.path.join(d, "*.csv")))

    csv_files = sorted(list(set(csv_files)))
    if not csv_files:
        raise FileNotFoundError(f"새별오름 883 기상 CSV 파일을 찾을 수 없습니다: {weather_dirs}")

    print(f"[*] Found {len(csv_files)} Saebyeol 883 weather CSV files:")
    for f in csv_files:
        print(f"    - {f}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except Exception:
            df = pd.read_csv(f, encoding="utf-8")
        dfs.append(df)

    raw_weather = pd.concat(dfs, ignore_index=True)

    # Column mapping
    col_mapping = {
        "지점": "station_id",
        "일시": "datetime",
        "기온(°C)": "aws_temperature",
        "풍향(deg)": "aws_wind_direction",
        "풍속(m/s)": "aws_wind_speed",
        "강수량(mm)": "aws_precipitation",
        "현지기압(hPa)": "aws_local_pressure",
        "해면기압(hPa)": "aws_sea_level_pressure",
        "습도(%)": "aws_humidity",
    }

    df_clean = raw_weather.rename(columns=col_mapping)
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"])
    df_clean = df_clean.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Select standard columns
    std_cols = [
        "datetime",
        "aws_temperature",
        "aws_wind_speed",
        "aws_wind_direction",
        "aws_humidity",
        "aws_local_pressure",
        "aws_sea_level_pressure",
        "aws_precipitation",
    ]
    avail_cols = [c for c in std_cols if c in df_clean.columns]
    df_clean = df_clean[avail_cols].copy()

    # Circular wind direction encoding
    if "aws_wind_direction" in df_clean.columns:
        sin_v, cos_v = wind_direction_to_sin_cos(df_clean["aws_wind_direction"])
        df_clean["aws_wind_dir_sin"] = sin_v
        df_clean["aws_wind_dir_cos"] = cos_v

    # Time-based linear interpolation for minor missing values (limit 6 consecutive hours)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear", limit=6)
    # Forward and backward fill for boundary values
    df_clean[numeric_cols] = df_clean[numeric_cols].bfill().ffill()

    # Preciptation fillna 0.0
    if "aws_precipitation" in df_clean.columns:
        df_clean["aws_precipitation"] = df_clean["aws_precipitation"].fillna(0.0)

    return df_clean


def build_and_save_master_dataset(
    gen_path: str = "data/interim/generation_hourly.parquet",
    output_path: str = "data/processed/merged_dataset.parquet",
    interim_weather_path: str = "data/interim/weather_hourly.parquet",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merges cleaned generation data with Saebyeol-Oreum 883 hourly weather and time features.
    """
    print("[*] Processing Saebyeol-Oreum AWS 883 weather...")
    df_weather = load_and_clean_saebyeol_weather()
    os.makedirs(os.path.dirname(interim_weather_path), exist_ok=True)
    df_weather.to_parquet(interim_weather_path, index=False)
    print(f"[+] Saved clean hourly weather ({len(df_weather):,} rows) to: {interim_weather_path}")

    print(f"[*] Loading generation data from: {gen_path}")
    df_gen = pd.read_parquet(gen_path)
    df_gen["datetime"] = pd.to_datetime(df_gen["datetime"])
    df_gen.sort_values("datetime", inplace=True)
    df_gen.reset_index(drop=True, inplace=True)

    # Merge
    df_merged = pd.merge(df_gen, df_weather, on="datetime", how="left")

    # Add cyclical time features
    df_merged = add_cyclical_time_features(df_merged, time_col="datetime")

    # Interpolate any residual boundary weather missings for generation timestamps
    weather_cols = [c for c in df_weather.columns if c != "datetime"]
    for wc in weather_cols:
        if wc in df_merged.columns:
            df_merged[wc] = df_merged[wc].interpolate(method="linear", limit=6).bfill().ffill()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_parquet(output_path, index=False)
    print(f"[+] Saved master merged dataset ({len(df_merged):,} rows) to: {output_path}")

    # Summary report
    qa_report = {
        "total_rows": len(df_merged),
        "start_time": str(df_merged["datetime"].min()),
        "end_time": str(df_merged["datetime"].max()),
        "weather_columns": weather_cols,
        "wind_speed_mean": float(df_merged["aws_wind_speed"].mean()),
        "wind_speed_max": float(df_merged["aws_wind_speed"].max()),
        "temperature_mean": float(df_merged["aws_temperature"].mean()),
    }

    return df_merged, qa_report


if __name__ == "__main__":
    df_merged, report = build_and_save_master_dataset()
    print("\n=== Master Merged Dataset QA Summary ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\nFirst 3 rows:")
    print(df_merged[["datetime", "generation_mwh", "aws_wind_speed", "aws_temperature", "aws_humidity"]].head(3))
