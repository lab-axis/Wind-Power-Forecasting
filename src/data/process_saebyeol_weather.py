"""
Saebyeol-Oreum (AWS 883) Hourly Weather Data Processing and Dataset Merging Module
- Loads raw hourly CSVs (2023, 2024, 2025)
- Standardizes meteorological column names
- Computes circular wind direction encodings (sin/cos)
- Restores an hourly grid and fills at most three past-observation hours; records missingness
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
    max_ffill_hours: int = 3,
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

    return clean_hourly_weather(df_clean, max_ffill_hours=max_ffill_hours)


def clean_hourly_weather(df_raw: pd.DataFrame, max_ffill_hours: int = 3) -> pd.DataFrame:
    """Causal cleaning on a regular grid. Flags preserve original observation status.

    The timestamp is assumed available immediately after its hourly interval ends.
    Real telemetry delays are unknown; this is an explicit offline protocol assumption.
    No extrapolation is made beyond the last timestamp in the source weather files.
    """
    if max_ffill_hours < 0:
        raise ValueError("max_ffill_hours must be nonnegative")
    df_clean = df_raw.copy()
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"])
    if df_clean.datetime.duplicated().any():
        raise ValueError("Duplicate weather timestamps")
    df_clean = df_clean.set_index("datetime").sort_index()
    grid = pd.date_range(df_clean.index.min(), df_clean.index.max(), freq="h", name="datetime")
    df_clean = df_clean.reindex(grid)
    for col in list(df_clean.columns):
        values = pd.to_numeric(df_clean[col], errors="coerce")
        observed = values.notna()
        last_time = pd.Series(grid, index=grid).where(observed).ffill()
        df_clean[col + "_is_missing"] = ~observed
        filled = values.ffill(limit=max_ffill_hours) if max_ffill_hours else values
        df_clean[col + "_is_imputed"] = (~observed) & filled.notna()
        df_clean[col + "_age_hours"] = (pd.Series(grid, index=grid) - last_time).dt.total_seconds() / 3600
        df_clean[col] = filled
    if "aws_wind_direction" in df_clean:
        sin_v, cos_v = wind_direction_to_sin_cos(df_clean["aws_wind_direction"])
        df_clean["aws_wind_dir_sin"] = sin_v
        df_clean["aws_wind_dir_cos"] = cos_v
    df_clean = df_clean.reset_index()

    return df_clean


def build_and_save_master_dataset(
    gen_path: str = "data/interim/generation_hourly.parquet",
    output_path: str = "data/processed/merged_dataset.parquet",
    interim_weather_path: str = "data/interim/weather_hourly.parquet",
    max_ffill_hours: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merges cleaned generation data with Saebyeol-Oreum 883 hourly weather and time features.
    """
    print("[*] Processing Saebyeol-Oreum AWS 883 weather...")
    df_weather = load_and_clean_saebyeol_weather(max_ffill_hours=max_ffill_hours)
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

    # Keep unavailable weather missing, including all of 2026 with current source files.
    weather_cols = [c for c in df_weather.columns if c != "datetime"]
    for col in [c for c in weather_cols if c.endswith("_is_missing")]:
        df_merged[col] = df_merged[col].fillna(True).astype(bool)
    for col in [c for c in weather_cols if c.endswith("_is_imputed")]:
        df_merged[col] = df_merged[col].fillna(False).astype(bool)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_parquet(output_path, index=False)
    print(f"[+] Saved master merged dataset ({len(df_merged):,} rows) to: {output_path}")

    # Summary report
    qa_report = {
        "total_rows": len(df_merged),
        "start_time": str(df_merged["datetime"].min()),
        "end_time": str(df_merged["datetime"].max()),
        "weather_columns": weather_cols,
        "max_ffill_hours": max_ffill_hours,
        "missing_values_by_column": df_merged.isna().sum().astype(int).to_dict(),
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
