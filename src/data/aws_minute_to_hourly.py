"""Reconstruct AWS hourly observations, not hourly averages of every variable.

Rules are audited against overlapping 2025 AWS files before any holdout scoring.
The wind window is (t-10min, t], with ten finite one-minute observations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

INSTANT = {'기온(°C)': 'aws_temperature', '습도(%)': 'aws_humidity',
           '현지기압(hPa)': 'aws_local_pressure', '해면기압(hPa)': 'aws_sea_level_pressure'}


def read_aws(path):
    frame = pd.read_csv(path, encoding='cp949')
    if not frame['지점'].eq(883).all():
        raise ValueError('Expected station 883 only')
    frame['일시'] = pd.to_datetime(frame['일시'])
    if frame['일시'].duplicated().any():
        raise ValueError('Duplicate source timestamps')
    return frame.set_index('일시').sort_index()


def reconstruct_hourly(frame):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Source timestamps must be unique and sorted')
    if not (frame.index == frame.index.floor('min')).all():
        raise ValueError('Expected minute-aligned source timestamps')
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq='min')
    source = frame.reindex(grid)
    hours = grid[grid.minute == 0]
    result = pd.DataFrame(index=hours)
    for raw, name in INSTANT.items():
        result[name] = source[raw].reindex(hours)
    speed = source['풍속(m/s)']
    direction = source['풍향(deg)']
    angle = np.deg2rad(direction)
    # Do not bridge missing minutes: min_periods=10 on the restored minute grid.
    mean_speed = speed.rolling(10, min_periods=10).mean()
    u = (speed * np.cos(angle)).rolling(10, min_periods=10).mean()
    v = (speed * np.sin(angle)).rolling(10, min_periods=10).mean()
    mean_direction = np.rad2deg(np.arctan2(v, u)) % 360
    # Hourly AWS publishes one decimal; calm wind has direction code zero.
    rounded_speed = np.floor(mean_speed * 10 + 0.5 + 1e-10) / 10
    mean_direction = mean_direction.mask(rounded_speed < 0.5, 0.0)
    result['aws_wind_speed'] = rounded_speed.reindex(hours)
    result['aws_wind_direction'] = (np.floor(mean_direction * 10 + 0.5 + 1e-10) / 10).reindex(hours)
    # Preserve a raw, unrounded diagnostic for the reconstruction audit.
    result['wind_speed_unrounded'] = mean_speed.reindex(hours)
    result['wind_direction_unrounded'] = mean_direction.reindex(hours)
    result['wind_minutes_present'] = speed.rolling(10, min_periods=1).count().reindex(hours)
    result['wind_vector_minutes_present'] = (speed.notna() & direction.notna()).astype(int).rolling(10, min_periods=1).sum().reindex(hours)
    result['source_timestamp_present'] = hours.isin(frame.index)
    result.index.name = 'datetime'
    return result.reset_index()


def load_minutes(paths):
    frames = [read_aws(Path(p)) for p in paths]
    merged = pd.concat(frames).sort_index()
    if merged.index.has_duplicates:
        raise ValueError('Overlapping minute files need explicit resolution')
    return merged
