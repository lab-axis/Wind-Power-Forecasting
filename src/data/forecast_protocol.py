"""Shared forecast protocol. Zero is an observed label, not a numerical epsilon."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
WEATHER_COLUMNS = [
    "aws_wind_speed", "aws_wind_dir_sin", "aws_wind_dir_cos",
    "aws_temperature", "aws_humidity", "aws_local_pressure",
]
ZERO_MWH = 0.0


def load_config():
    return yaml.safe_load((ROOT / "configs/base_config.yaml").read_text(encoding="utf-8"))


def validate_hourly_frame(frame):
    times = pd.DatetimeIndex(frame["datetime"])
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted")
    if len(times) > 1 and not (np.diff(times.asi8) == pd.Timedelta(hours=1).value).all():
        # Convert explicitly: pandas can store timestamps in microseconds.
        if not (times.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
            raise ValueError("Missing timestamp rows: restore the hourly grid before windowing")


def valid_window_indices(data, lookback, horizon):
    """Origins are exclusive: input [i-L:i], target i+h-1. Never compress gaps."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    return [i for i in range(lookback, len(data) - horizon + 1)
            if np.isfinite(data[i-lookback:i]).all()
            and np.isfinite(data[i+horizon-1, 0])]


def prepare_splits(frame, config=None):
    config = config or load_config()
    validate_hourly_frame(frame)
    splits = config["data_splits"]
    masks = {name: frame.datetime.between(splits[name]["start"], splits[name]["end"])
             for name in ("train", "validation", "test")}
    mean = frame.loc[masks["train"], WEATHER_COLUMNS].mean()
    std = frame.loc[masks["train"], WEATHER_COLUMNS].std().replace(0, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Insufficient finite training weather observations")
    normalized = frame.copy()
    normalized[WEATHER_COLUMNS] = (frame[WEATHER_COLUMNS] - mean) / std
    raw = {k: frame.loc[v].reset_index(drop=True) for k, v in masks.items()}
    arrays = {k: normalized.loc[v, ["generation_mwh"] + WEATHER_COLUMNS].to_numpy()
              for k, v in masks.items()}
    scaler = {"mean": mean.to_dict(), "std": std.to_dict(), "fit_split": "train"}
    return raw, arrays, scaler


def prediction_frame(split, indices, horizon, y_true, y_pred_raw, **extra):
    target = np.asarray(indices, dtype=int) + horizon - 1
    origin = np.asarray(indices, dtype=int) - 1
    frame = pd.DataFrame({
        "target_time": split.datetime.iloc[target].to_numpy(),
        "issue_time": split.datetime.iloc[origin].to_numpy(),
        "y_true": np.asarray(y_true).ravel(),
        "y_pred_raw": np.asarray(y_pred_raw).ravel(),
    })
    for key, value in extra.items():
        frame[key] = np.asarray(value).ravel()
    return frame


def align_predictions(frames):
    """Return exact common target timestamps and verify all reference labels."""
    common = None
    for name, f in frames.items():
        if f.target_time.duplicated().any():
            raise ValueError(f"Duplicate target timestamps: {name}")
        if not np.isfinite(f[["y_true", "y_pred_raw"]].to_numpy()).all():
            raise ValueError(f"Nonfinite predictions: {name}")
        idx = pd.DatetimeIndex(f.target_time)
        common = idx if common is None else common.intersection(idx)
    if common is None or common.empty:
        raise ValueError("No common evaluation samples")
    common = common.sort_values()
    aligned = {k: f.set_index("target_time").loc[common].reset_index() for k, f in frames.items()}
    reference = next(iter(aligned.values()))
    for name, f in aligned.items():
        if not np.allclose(f.y_true, reference.y_true, atol=3e-6, rtol=1e-6):
            raise ValueError(f"Target label mismatch: {name}")
        if not f.issue_time.equals(reference.issue_time):
            raise ValueError(f"Forecast issue timestamp mismatch: {name}")
    return aligned
