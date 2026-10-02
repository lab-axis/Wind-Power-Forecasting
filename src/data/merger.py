"""
발전량 시계열과 기상 관측자료 병합 및 QA 모듈
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


def merge_generation_and_weather(
    df_gen: pd.DataFrame,
    df_weather: Optional[pd.DataFrame] = None,
    time_col: str = "datetime",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    발전량 데이터와 기상 데이터를 datetime 기준으로 병합하고 QA 메트릭 산출
    - df_weather가 None이거나 비어있을 경우 발전량 단독 데이터셋 구성 (M1 실험군 지원)
    """
    df_g = df_gen.copy()
    df_g[time_col] = pd.to_datetime(df_g[time_col])

    if df_weather is not None and not df_weather.empty:
        df_w = df_weather.copy()
        df_w[time_col] = pd.to_datetime(df_w[time_col])
        df_merged = pd.merge(df_g, df_w, on=time_col, how="left")
        weather_matched = df_merged[df_w.columns.difference([time_col])].notna().any(axis=1).sum()
        weather_match_rate = float(round(weather_matched / len(df_merged) * 100, 2))
    else:
        df_merged = df_g
        weather_match_rate = 0.0

    # QA 메트릭 산출
    total_rows = len(df_merged)
    gen_valid = df_merged["generation_mwh"].notna().sum()
    gen_null = df_merged["generation_mwh"].isna().sum()

    missing_by_col = df_merged.isnull().sum().to_dict()
    missing_pct_by_col = {
        k: float(round(v / total_rows * 100, 2)) for k, v in missing_by_col.items()
    }

    qa_report = {
        "total_rows": int(total_rows),
        "start_time": str(df_merged[time_col].min()),
        "end_time": str(df_merged[time_col].max()),
        "generation_valid_count": int(gen_valid),
        "generation_null_count": int(gen_null),
        "weather_data_present": bool(df_weather is not None and not df_weather.empty),
        "weather_match_rate_pct": weather_match_rate,
        "missing_pct_by_col": missing_pct_by_col,
    }

    return df_merged, qa_report


def build_and_save_merged_dataset(
    gen_parquet_path: str = "data/interim/generation_hourly.parquet",
    weather_parquet_path: Optional[str] = None,
    output_parquet_path: str = "data/processed/merged_dataset.parquet",
    output_qa_path: str = "reports/tables/merged_qa_summary.csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    최종 정합 데이터셋 생성 및 저장 파이프라인
    """
    if not os.path.exists(gen_parquet_path):
        from src.data.load_generation import build_and_save_generation_interim
        build_and_save_generation_interim(output_parquet=gen_parquet_path)

    df_gen = pd.read_parquet(gen_parquet_path)

    df_weather = None
    if weather_parquet_path and os.path.exists(weather_parquet_path):
        df_weather = pd.read_parquet(weather_parquet_path)

    df_merged, qa_report = merge_generation_and_weather(df_gen, df_weather)

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_qa_path), exist_ok=True)

    df_merged.to_parquet(output_parquet_path, index=False)

    df_qa = pd.DataFrame(
        [
            {"metric": k, "value": str(v)}
            for k, v in qa_report.items()
            if k != "missing_pct_by_col"
        ]
    )
    for col, pct in qa_report["missing_pct_by_col"].items():
        df_qa = pd.concat(
            [df_qa, pd.DataFrame([{"metric": f"missing_pct_{col}", "value": f"{pct}%"}])],
            ignore_index=True,
        )

    df_qa.to_csv(output_qa_path, index=False, encoding="utf-8-sig")

    return df_merged, qa_report


if __name__ == "__main__":
    df, report = build_and_save_merged_dataset()
    print("=== Dataset Merge and QA Complete ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
