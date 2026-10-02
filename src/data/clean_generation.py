"""
상명풍력 발전량 데이터 품질 분석, 0값 EDA 및 sanity check 모듈
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_yearly_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 및 월별 발전량 통계 계산 (MWh)"""
    df_calc = df.copy()
    df_calc["year"] = df_calc["datetime"].dt.year
    df_calc["month"] = df_calc["datetime"].dt.month
    df_calc["year_month"] = df_calc["datetime"].dt.to_period("M")

    # 월별 합계 및 평균
    monthly = (
        df_calc.groupby("year_month")
        .agg(
            total_mwh=("generation_mwh", "sum"),
            mean_mwh=("generation_mwh", "mean"),
            max_mwh=("generation_mwh", "max"),
            valid_hours=("generation_mwh", "count"),
            zero_hours=("generation_mwh", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )
    monthly["zero_ratio_pct"] = (
        monthly["zero_hours"] / monthly["valid_hours"] * 100.0
    ).round(2)
    monthly["total_mwh"] = monthly["total_mwh"].round(2)
    monthly["mean_mwh"] = monthly["mean_mwh"].round(4)
    monthly["max_mwh"] = monthly["max_mwh"].round(4)

    return monthly


def compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 발전량 합계 및 가동률 추정"""
    df_calc = df.copy()
    df_calc["year"] = df_calc["datetime"].dt.year

    yearly = (
        df_calc.groupby("year")
        .agg(
            total_mwh=("generation_mwh", "sum"),
            mean_mwh=("generation_mwh", "mean"),
            max_mwh=("generation_mwh", "max"),
            valid_hours=("generation_mwh", "count"),
            zero_hours=("generation_mwh", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    # 21MW 기준 이용률 (Capacity Factor = Total Generation / (21MW * valid_hours))
    yearly["capacity_factor_pct"] = (
        yearly["total_mwh"] / (21.0 * yearly["valid_hours"]) * 100.0
    ).round(2)
    yearly["zero_ratio_pct"] = (
        yearly["zero_hours"] / yearly["valid_hours"] * 100.0
    ).round(2)
    yearly["total_mwh"] = yearly["total_mwh"].round(2)
    yearly["mean_mwh"] = yearly["mean_mwh"].round(4)

    return yearly


def analyze_zero_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    0값 연속 구간(Zero Streak) 길이 분포 분석
    """
    df_sorted = df.sort_values("datetime").copy()
    is_zero = (df_sorted["generation_mwh"] == 0.0).astype(int)

    # 연속된 0 그룹 ID 생성
    streak_id = (is_zero != is_zero.shift()).cumsum()
    streak_id[is_zero == 0] = np.nan

    streaks = []
    for sid, group in df_sorted.groupby(streak_id):
        streaks.append(
            {
                "start_time": group["datetime"].iloc[0],
                "end_time": group["datetime"].iloc[-1],
                "duration_hours": len(group),
            }
        )

    if not streaks:
        return pd.DataFrame()

    df_streaks = pd.DataFrame(streaks)
    df_streaks.sort_values(by="duration_hours", ascending=False, inplace=True)
    df_streaks.reset_index(drop=True, inplace=True)
    return df_streaks


def find_all_zero_days(df: pd.DataFrame) -> pd.DataFrame:
    """하루 24시간 전체가 0 발전량인 날짜 탐색 (정비/고장/curtailment 의심)"""
    df_calc = df.copy()
    df_calc["date"] = df_calc["datetime"].dt.date
    daily = (
        df_calc.groupby("date")
        .agg(
            valid_hours=("generation_mwh", "count"),
            zero_hours=("generation_mwh", lambda x: (x == 0).sum()),
            sum_mwh=("generation_mwh", "sum"),
        )
        .reset_index()
    )
    all_zero_days = daily[(daily["valid_hours"] >= 20) & (daily["sum_mwh"] == 0.0)]
    return all_zero_days


def generate_eda_reports_and_plots(
    parquet_path: str = "data/interim/generation_hourly.parquet",
    output_dir_tables: str = "reports/tables",
    output_dir_figures: str = "reports/figures",
):
    """
    전체 발전량 데이터 분석 보고서 표 및 플롯 생성
    """
    os.makedirs(output_dir_tables, exist_ok=True)
    os.makedirs(output_dir_figures, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. 연도별 / 월별 통계
    yearly_df = compute_yearly_summary(df)
    monthly_df = compute_yearly_monthly_stats(df)
    yearly_df.to_csv(os.path.join(output_dir_tables, "generation_yearly_summary.csv"), index=False, encoding="utf-8-sig")
    monthly_df.to_csv(os.path.join(output_dir_tables, "generation_monthly_summary.csv"), index=False, encoding="utf-8-sig")

    # 2. 0값 연속 구간 분석
    streaks_df = analyze_zero_streaks(df)
    if not streaks_df.empty:
        streaks_df.to_csv(os.path.join(output_dir_tables, "zero_streak_summary.csv"), index=False, encoding="utf-8-sig")

    # 3. 하루 전체 0인 날짜
    all_zero_days = find_all_zero_days(df)
    all_zero_days.to_csv(os.path.join(output_dir_tables, "all_zero_days.csv"), index=False, encoding="utf-8-sig")

    # 4. 시각화 (matplotlib)
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # (a) 전체 시계열 개요 (2025-02-01 단위 전환선 표시)
    axes[0].plot(df["datetime"], df["generation_mwh"], color="#1f77b4", lw=0.6, alpha=0.8, label="Hourly Generation (MWh)")
    axes[0].axvline(pd.to_datetime("2025-02-01"), color="red", linestyle="--", lw=1.5, label="Scale Transition (2025-02-01)")
    axes[0].axhline(21.0, color="gray", linestyle=":", lw=1.2, label="Rated Capacity (21 MW)")
    axes[0].set_title("Sangmyeong Wind Power Generation (Hourly MWh, 2023 - 2026)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Generation (MWh)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # (b) 월별 총 발전량 막대 그래프
    monthly_df["ym_str"] = monthly_df["year_month"].astype(str)
    axes[1].bar(monthly_df["ym_str"], monthly_df["total_mwh"], color="#2ca02c", alpha=0.85)
    axes[1].set_title("Monthly Total Generation (MWh)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Total Generation (MWh)")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(True, alpha=0.3, axis="y")

    # (c) 시간대별(0~23시) 평균 발전량 및 0 비율
    df["hour_plot"] = df["datetime"].dt.hour
    hourly_stat = df.groupby("hour_plot").agg(
        mean_mwh=("generation_mwh", "mean"),
        zero_pct=("generation_mwh", lambda x: (x == 0).sum() / len(x) * 100.0)
    ).reset_index()

    ax2_twin = axes[2].twinx()
    axes[2].plot(hourly_stat["hour_plot"], hourly_stat["mean_mwh"], marker="o", color="#d62728", lw=2, label="Mean Generation (MWh)")
    ax2_twin.bar(hourly_stat["hour_plot"], hourly_stat["zero_pct"], alpha=0.3, color="#17becf", label="Zero Ratio (%)")
    axes[2].set_title("Diurnal Pattern: Mean Generation & Zero Ratio by Hour of Day", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Hour of Day (0 - 23)")
    axes[2].set_ylabel("Mean Generation (MWh)", color="#d62728")
    ax2_twin.set_ylabel("Zero Ratio (%)", color="#17becf")
    axes[2].set_xticks(range(24))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_figures, "generation_eda_overview.png"), dpi=200)
    plt.close()

    print("EDA Reports and figures successfully generated.")
    return yearly_df, monthly_df, streaks_df, all_zero_days


if __name__ == "__main__":
    y_df, m_df, s_df, z_days = generate_eda_reports_and_plots()
    print("\n=== Yearly Summary ===")
    print(y_df)
    print(f"\nTop 5 Longest Zero Streaks (hours):")
    print(s_df.head(5))
    print(f"\nTotal All-zero Days: {len(z_days)}")
