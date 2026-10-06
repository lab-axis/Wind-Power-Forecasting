"""Validate weather-only reconstruction on 2025; stage 2026 weather separately."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.forecast_protocol import ROOT
from src.data.aws_minute_to_hourly import INSTANT, load_minutes, read_aws, reconstruct_hourly
from src.experiments.run_comprehensive_extended_matrix import sha256


def main():
    base = ROOT/'data/raw/weather'
    minute = next((base/'saebyeol_883-minute').glob('*2025-01*'))
    hourly = next((base/'saebyeol_883-hour').glob('*2025_2025*'))
    reference = read_aws(hourly)
    candidate = reconstruct_hourly(load_minutes([minute])).set_index('datetime')
    joined = candidate.join(reference, how='inner')
    rows = []
    mapping = dict(INSTANT, **{'풍속(m/s)': 'aws_wind_speed', '풍향(deg)': 'aws_wind_direction'})
    for raw, name in mapping.items():
        predicted = joined[name]
        error = predicted - joined[raw]
        if name == 'aws_wind_direction':
            error = (error + 180) % 360 - 180
        e = error.dropna().abs()
        tolerance = 0.100001 if 'wind' in name else 1e-9
        rows.append(dict(variable=name, paired_count=len(e), reference_count=int(joined[raw].notna().sum()),
                         mae=float(e.mean()), max_absolute_error=float(e.max()),
                         tolerance=tolerance, within_tolerance=int(e.le(tolerance).sum()),
                         exact_to_1e_9=int(e.le(1e-9).sum())))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT/'reports/tables/aws_2025_reconstruction_audit.csv', index=False)
    # Rounded values may differ by one published least-significant digit.
    passed = bool((table.paired_count > 8000).all() and (table.paired_count == table.within_tolerance).all())
    paths = sorted((base/'saebyeol_883-minute').glob('*2026-*.csv'))
    record = dict(status='passed' if passed else 'needs_review', reference_year=2025,
                  generation_used=False, forecast_scores_computed=False,
                  rules=dict(instant=list(INSTANT.values()), wind='trailing ten finite minutes, speed-weighted circular direction',
                             calm='published rounded speed < 0.5 m/s gives direction zero',
                             rounding='nearest tenth, half up', imputation='none',
                             precipitation='not reconstructed; not a model input'),
                  source_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in [minute, hourly]+paths},
                  implementation_sha256=sha256(ROOT/'src/data/aws_minute_to_hourly.py'),
                  audit_script_sha256=sha256(Path(__file__)), variables=rows,
                  official_definition='https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=239')
    if passed:
        staged = reconstruct_hourly(load_minutes(paths))
        staged = staged[staged.datetime.between('2026-01-01 01:00', '2026-03-31 00:00')]
        destination = ROOT/'.experiment_archive/holdout_2026/weather_hourly_candidate.parquet'
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged.to_parquet(destination, index=False)
        record['staged_weather'] = dict(path=destination.relative_to(ROOT).as_posix(), sha256=sha256(destination),
                                       rows=len(staged), missing_by_column=staged.isna().sum().to_dict(),
                                       status='candidate_not_merged_or_used_for_forecasting')
    (ROOT/'reports/aws_reconstruction_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(table.to_string(index=False))
    print(record['status'], record.get('staged_weather', {}))
    if not passed:
        raise RuntimeError('Weather audit requires review; no 2026 candidate exported')


if __name__ == '__main__':
    main()
