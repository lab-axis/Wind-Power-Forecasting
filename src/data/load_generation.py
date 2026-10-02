"""
상명풍력 발전실적 데이터 로딩 및 단위 정규화 모듈

주요 기능:
1. 상명풍력 시간별 발전실적 CSV 로딩 (cp949 / utf-8 자동 감지)
2. Wide 포맷 (기준일, 1시~24시) -> Hourly Long 포맷 (datetime, generation_mwh) 변환
3. 24시 timestamp를 다음날 00:00으로 변환 (date + 1 day 00:00)
4. 2025-02-01 전후 스케일 불연속성(Wh -> kWh)을 물리적 설비용량(21 MW)에 기반하여 MWh로 정규화
5. 원본 raw 값(raw_generation)과 보정값(generation_mwh) 모두 보존
6. 물리적 상한(21 MWh) 및 음수값 검증
7. 0값 보존 (0을 임의로 NaN 변환하지 않음)
"""

import os
import glob
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


def detect_file_encoding(file_path: str) -> str:
    """한글 인코딩 시도 (cp949, euc-kr, utf-8-sig, utf-8)"""
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "cp949"


def load_raw_generation_csv(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    상명풍력 원본 CSV 로드
    file_path가 주어지지 않으면 기본 경로 및 파일 패턴으로 탐색
    """
    if file_path is None or not os.path.exists(file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        candidates = [
            os.path.join(project_root, "data/raw/generation/한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv"),
            "data/raw/generation/한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv",
            "../data/raw/generation/한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv",
            "한국중부발전(주)_(AI 친화 데이터)풍력 발전 실적(상명풍력)_20260330.csv",
        ]
        # glob 탐색
        candidates.extend(glob.glob(os.path.join(project_root, "data/raw/generation/*상명풍력*.csv")))
        candidates.extend(glob.glob("data/raw/generation/*상명풍력*.csv"))
        candidates.extend(glob.glob("../data/raw/generation/*상명풍력*.csv"))
        candidates.extend(glob.glob("*상명풍력*.csv"))

        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(f"상명풍력 원본 CSV 파일을 찾을 수 없습니다: {file_path}")

    enc = detect_file_encoding(file_path)
    df = pd.read_csv(file_path, encoding=enc)

    # 빈 행 제거 (전체 컬럼 NaN인 행)
    df = df.dropna(how="all").copy()

    # 기준일 컬럼이 NaN인 행 제거
    if "기준일" in df.columns:
        df = df.dropna(subset=["기준일"]).copy()
    else:
        # 첫 번째 컬럼이 기준일일 경우
        df = df.dropna(subset=[df.columns[0]]).copy()
        df.rename(columns={df.columns[0]: "기준일"}, inplace=True)

    return df


def wide_to_hourly_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    기준일 + 1시~24시 컬럼 구조를 datetime, raw_generation 형태의 Long format으로 변환.
    24시는 다음 날 00:00으로 변환됨.
    """
    hour_cols = [f"{h}시" for h in range(1, 25)]
    # 존재하는 시간 컬럼 확인
    valid_hour_cols = [c for c in hour_cols if c in df_wide.columns]
    if len(valid_hour_cols) != 24:
        raise ValueError(f"24개 시간 컬럼 중 일부가 누락되었습니다: {valid_hour_cols}")

    melted = pd.melt(
        df_wide,
        id_vars=["기준일"],
        value_vars=valid_hour_cols,
        var_name="hour_str",
        value_name="raw_generation",
    )

    # 시간 숫자 추출 (1 ~ 24)
    melted["hour"] = melted["hour_str"].str.extract(r"(\d+)").astype(int)

    # 기준일 파싱
    melted["base_date"] = pd.to_datetime(melted["기준일"])

    # 1시~23시는 base_date + (hour)시간, 24시는 base_date + 1일 00:00
    # 예: 2023-01-01 1시 -> 2023-01-01 01:00:00
    #     2023-01-01 24시 -> 2023-01-02 00:00:00
    melted["datetime"] = melted.apply(
        lambda row: row["base_date"] + pd.Timedelta(hours=row["hour"]), axis=1
    )

    df_long = melted[["datetime", "base_date", "hour", "raw_generation"]].copy()
    df_long.sort_values(by="datetime", inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    return df_long


def normalize_generation_units(
    df_long: pd.DataFrame,
    scale_split_date: str = "2025-02-01",
    rated_capacity_mw: float = 21.0,
) -> pd.DataFrame:
    """
    2025-02-01을 기준으로 스케일 정규화 (Wh/kWh -> MWh)
    - 2025-02-01 00:00:00 (즉, 2025-01-31 24시)까지: raw / 1,000,000 (Wh -> MWh)
    - 2025-02-01 01:00:00 (즉, 2025-02-01 1시)부터: raw / 1,000 (kWh -> MWh)
    - 원시값 raw_generation과 보정값 generation_mwh를 함께 유지
    """
    df = df_long.copy()
    
    # base_date가 2025-02-01 이전이면 Wh, 2025-02-01부터는 kWh
    # datetime 기준으로 보면 2025-02-01 00:00:00 이하가 Wh, 2025-02-01 01:00:00 이상이 kWh
    split_dt = pd.to_datetime(f"{scale_split_date} 00:00:00")
    if "base_date" in df.columns:
        before_mask = df["base_date"] < pd.to_datetime(scale_split_date)
    else:
        before_mask = df["datetime"] <= split_dt
    after_mask = ~before_mask

    df["generation_mwh"] = np.nan
    df.loc[before_mask, "generation_mwh"] = (
        df.loc[before_mask, "raw_generation"] / 1_000_000.0
    )
    df.loc[after_mask, "generation_mwh"] = (
        df.loc[after_mask, "raw_generation"] / 1_000.0
    )

    # 단위 보정 추정 플래그 기록
    df["scale_unit_assumed"] = "Wh"
    df.loc[after_mask, "scale_unit_assumed"] = "kWh"

    # 물리적 검증 플래그
    df["is_capacity_exceeded"] = df["generation_mwh"] > rated_capacity_mw
    df["is_negative"] = df["generation_mwh"] < 0.0
    df["is_zero"] = df["generation_mwh"] == 0.0

    return df


def validate_generation_data(
    df: pd.DataFrame, rated_capacity_mw: float = 21.0
) -> Dict[str, Any]:
    """
    발전 데이터 품질 및 정합성 검증 보고서 딕셔너리 생성
    """
    total_hours = len(df)
    valid_count = df["generation_mwh"].notna().sum()
    null_count = df["generation_mwh"].isna().sum()
    zero_count = (df["generation_mwh"] == 0.0).sum()
    zero_ratio = (zero_count / valid_count * 100.0) if valid_count > 0 else 0.0

    min_val = df["generation_mwh"].min()
    max_val = df["generation_mwh"].max()
    mean_val = df["generation_mwh"].mean()

    exceeded_count = (df["generation_mwh"] > rated_capacity_mw).sum()
    negative_count = (df["generation_mwh"] < 0.0).sum()

    dt_min = df["datetime"].min()
    dt_max = df["datetime"].max()

    report = {
        "start_datetime": str(dt_min),
        "end_datetime": str(dt_max),
        "total_records": int(total_hours),
        "valid_records": int(valid_count),
        "null_records": int(null_count),
        "null_ratio_pct": float(round(null_count / total_hours * 100, 3)),
        "zero_records": int(zero_count),
        "zero_ratio_pct": float(round(zero_ratio, 2)),
        "min_mwh": float(round(min_val, 4)) if pd.notna(min_val) else None,
        "max_mwh": float(round(max_val, 4)) if pd.notna(max_val) else None,
        "mean_mwh": float(round(mean_val, 4)) if pd.notna(mean_val) else None,
        "capacity_exceeded_count": int(exceeded_count),
        "negative_count": int(negative_count),
    }

    return report


def build_and_save_generation_interim(
    raw_file_path: Optional[str] = None,
    output_parquet: str = "data/interim/generation_hourly.parquet",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    상명풍력 원본 CSV를 읽고 변환 및 정규화하여 parquet 파일로 저장하는 전체 실행 파이프라인
    """
    df_wide = load_raw_generation_csv(raw_file_path)
    df_long = wide_to_hourly_long(df_wide)
    df_normalized = normalize_generation_units(df_long)
    report = validate_generation_data(df_normalized)

    # 출력 디렉터리 확인
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_normalized.to_parquet(output_parquet, index=False)

    return df_normalized, report


if __name__ == "__main__":
    df, report = build_and_save_generation_interim()
    print("=== Sangmyeong Generation Preprocessing Complete ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
