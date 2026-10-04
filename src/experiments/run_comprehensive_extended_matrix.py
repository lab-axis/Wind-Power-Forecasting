"""Corrected v3 benchmark: train afresh, preserve raw predictions, align by target time.

Run from the repository root: python -m src.experiments.run_comprehensive_extended_matrix
2025 is a previously consulted development benchmark, not a blind holdout.
This historical recipe retains validation-loss selection. For the current paper
comparison with uniform CRPS selection use run_multiseed_crps instead.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import torch

from src.data.forecast_protocol import (
    ROOT, WEATHER_COLUMNS, load_config, prepare_splits, prediction_frame, align_predictions,
)
from src.data.load_generation import build_and_save_generation_interim
from src.data.process_saebyeol_weather import build_and_save_master_dataset
from src.features.lag_features import create_lag_features, create_rolling_features
from src.features.time_features import add_cyclical_time_features
from src.models.emfn_trainer import train_and_evaluate_emfn
from src.models.torch_trainer import train_and_evaluate_torch_model, WindTimeSeriesDataset
from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.models.dlinear import DLinearForecaster
from src.models.rnn import LSTMForecaster, GRUForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster
from src.models.hurdle_beta import evaluate_zero_head
from src.utils.metrics import (evaluate_wind_forecast, compute_crps_point,
    compute_crps_hurdle_beta, hurdle_beta_quantiles, compute_prediction_interval_metrics)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Seeded CUDA runs; bitwise reproducibility is not promised across GPU kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def slug(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def tabular_features(frame):
    # Explicit allowlist prevents audit flags, labels or raw unit columns becoming features.
    columns = ["datetime", "generation_mwh"] + WEATHER_COLUMNS
    f = add_cyclical_time_features(frame[columns].copy())
    f = create_lag_features(f, ["generation_mwh"], [1, 2, 3, 6, 12, 24, 48, 168])
    f = create_rolling_features(f, ["generation_mwh"], [3, 6, 24])
    f = create_lag_features(f, ["aws_wind_speed", "aws_temperature", "aws_local_pressure"], [1, 2, 3, 6, 24])
    f = create_rolling_features(f, ["aws_wind_speed"], [3, 6, 24])
    return f, [c for c in f if c != "datetime"]


def score_predictions(frame, capacity=21.0, grid=1001):
    yt, yp = frame.y_true.to_numpy(), frame.y_pred_raw.to_numpy()
    m = evaluate_wind_forecast(yt, yp, y_pred_raw=yp,
        wind_speed=frame.wind_speed_target.to_numpy(), rated_capacity_mwh=capacity,
        timestamps=frame.target_time)
    m["zero_point_f1"] = m.pop("zero_f1")
    m["zero_point_balanced_acc"] = m.pop("zero_balanced_acc")
    if "p_pos" in frame:
        zero = evaluate_zero_head(yt, frame.p_pos.to_numpy())
        m.update(zero_auroc=zero["auroc"], zero_auprc=zero["auprc_zero"],
            zero_brier=zero["brier_score_zero"], zero_ece=zero["ece_zero"],
            zero_head_f1=zero["f1_zero"])
        m["mae_median"] = evaluate_wind_forecast(yt, frame.y_median, rated_capacity_mwh=capacity)["mae"]
        m["median_gain_pct"] = 100 * (m["mae"] - m["mae_median"]) / m["mae"]
        p, a, b = frame.p_pos.to_numpy(), frame.alpha.to_numpy(), frame.beta.to_numpy()
        m["crps"] = compute_crps_hurdle_beta(yt, p, a, b, capacity, n_eval_points=grid)
        lower = hurdle_beta_quantiles(p, a, b, .05, capacity)
        upper = hurdle_beta_quantiles(p, a, b, .95, capacity)
        m.update(compute_prediction_interval_metrics(yt, lower, upper, capacity))
    else:
        m["crps"] = compute_crps_point(yt, np.clip(yp, 0, capacity))
    return m


def add_observations(f, test, capacity):
    lookup = test.set_index("datetime")
    # Exact common reference values, not model-specific float32 round trips.
    f["y_true"] = lookup.loc[f.target_time, "generation_mwh"].to_numpy()
    ws = lookup.aws_wind_speed.where(~lookup.aws_wind_speed_is_missing)
    f["wind_speed_target"] = ws.loc[f.target_time].to_numpy()
    f["y_pred_clipped"] = np.clip(f.y_pred_raw, 0, capacity)
    return f


def summarize_data(frame):
    tables = ROOT / "reports/tables"
    rows = []
    for label, d in [("file_all", frame), ("study_2023_2025", frame[frame.datetime.dt.year <= 2025]),
                      ("test_2025", frame[frame.datetime.dt.year == 2025])]:
        y = d.generation_mwh
        rows.append(dict(scope=label, rows=len(d), valid=int(y.notna().sum()), missing=int(y.isna().sum()),
            exact_zero=int(y.eq(0).sum()), zero_pct=100*y.eq(0).sum()/y.notna().sum(),
            min_positive=y[y>0].min(), weather_missing_rows=int(d[WEATHER_COLUMNS].isna().any(axis=1).sum())))
    pd.DataFrame(rows).to_csv(tables / "data_quality_by_scope.csv", index=False)
    missing = frame.isna().sum()
    pd.DataFrame({"metric": ["total_rows", "generation_null_count", "weather_data_present"] + ["missing_count_"+c for c in missing.index],
                  "value": [len(frame), int(frame.generation_mwh.isna().sum()), True] + missing.tolist()}).to_csv(tables / "merged_qa_summary.csv", index=False)
    # Exclude 2026 and all imputed target-time wind speeds from physical association EDA.
    d = frame[(frame.datetime.dt.year <= 2025) & frame.generation_mwh.notna() & ~frame.aws_wind_speed_is_missing].copy()
    d["wind_bin"] = pd.cut(d.aws_wind_speed, [0,2,4,6,8,10,30], right=False)
    bins = d.groupby("wind_bin", observed=False).agg(total_hours=("generation_mwh", "count"),
        zero_hours=("generation_mwh", lambda s:s.eq(0).sum()), mean_generation_mwh=("generation_mwh","mean"),
        max_generation_mwh=("generation_mwh","max")).reset_index()
    bins["zero_probability"] = bins.zero_hours/bins.total_hours
    bins.to_csv(tables / "zero_wind_bin_analysis.csv", index=False)


def run_full_pipeline(horizons=None):
    config = load_config()
    cfg = config["protocol"]
    seed = cfg["seeds"][0]
    if len(cfg["seeds"]) != 1:
        raise ValueError("This benchmark entry point records one seed per run; execute separate runs for additional seeds")
    torch.set_num_threads(cfg["torch_threads"])
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = horizons or list(config["modeling"]["forecast_horizons"].values())
    capacity = config["plant_specs"]["max_hourly_mwh"]
    L = config["modeling"]["lookback_hours"]
    run_id = "corrected_v3_" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%dT%H%M%S%z")
    run_dir = ROOT / "models/runs" / run_id
    run_dir.mkdir(parents=True)
    source_paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").glob("*.yaml"))
    source_paths += [ROOT / "requirements.txt", ROOT / "environment.yml"]
    source_hashes = {p.relative_to(ROOT).as_posix():sha256(p) for p in source_paths}
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in source_paths: archive.write(p, p.relative_to(ROOT).as_posix())
    shutil.copy2(ROOT / "configs/base_config.yaml", run_dir / "config.yaml")
    manifest = dict(run_id=run_id, status="running", protocol=cfg, source_hashes=source_hashes,
        raw_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT / "data/raw").rglob("*.csv")},
        environment={p:importlib.metadata.version(p) for p in ["torch","numpy","pandas","scipy","scikit-learn","lightgbm","pyarrow"]},
        device=device, gpu=torch.cuda.get_device_name(0) if device=="cuda" else None,
        test_role=cfg["test_role"], horizons=horizons, experiments=[])
    def save_manifest():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest()
    print(f"RUN {run_id} | {device} | seed {seed}", flush=True)
    build_and_save_generation_interim(output_parquet=str(ROOT / "data/interim/generation_hourly.parquet"))
    frame, _ = build_and_save_master_dataset(gen_path=str(ROOT / "data/interim/generation_hourly.parquet"),
        output_path=str(ROOT / "data/processed/merged_dataset.parquet"),
        interim_weather_path=str(ROOT / "data/interim/weather_hourly.parquet"), max_ffill_hours=cfg["max_ffill_hours"])
    manifest["processed_data_sha256"] = sha256(ROOT / "data/processed/merged_dataset.parquet")
    raw, arrays, scaler = prepare_splits(frame, config)
    manifest["weather_scaler"] = scaler
    tab, feature_cols = tabular_features(frame)
    tab_splits = {k:tab[tab.datetime.between(config["data_splits"][k]["start"], config["data_splits"][k]["end"])].reset_index(drop=True) for k in raw}
    manifest["features"] = {"sequence": ["generation_mwh"] + WEATHER_COLUMNS, "lightgbm":feature_cols}
    save_manifest()
    summarize_data(frame)
    factories = {
        "DLinear (+Weather)": lambda: DLinearForecaster(lookback_steps=L, forecast_horizon=1, input_dim=7),
        "LSTM (+Weather)": lambda: LSTMForecaster(input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=1),
        "GRU (+Weather)": lambda: GRUForecaster(input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=1),
        "CNN-LSTM (+Weather)": lambda: CNNLSTMForecaster(input_dim=7, lookback_steps=L, forecast_horizon=1),
        "TCN (+Weather)": lambda: TCNForecaster(input_dim=7, num_channels=[64]*3, kernel_size=3, forecast_horizon=1),
        "Transformer (+Weather)": lambda: TimeSeriesTransformerForecaster(input_dim=7, lookback_steps=L, forecast_horizon=1),
    }
    rows, coverage = [], []
    for h in horizons:
        frames, timings = {}, {}
        def record(name, f, info, checkpoint=None):
            f = add_observations(f, raw["test"], capacity)
            dest = run_dir / f"{slug(name)}_{h}h_predictions.parquet"
            f.to_parquet(dest, index=False)
            frames[name] = f
            timings[name] = info.get("train_time_sec", 0.0)
            record = dict(model=name, horizon=h, seed=seed, prediction_file=dest.name,
                prediction_sha256=sha256(dest), checkpoint=checkpoint, **info)
            if checkpoint: record["checkpoint_sha256"] = sha256(run_dir / checkpoint)
            manifest["experiments"].append(record)
            save_manifest()
            print(f"  +{h}h {name}: trained in {timings[name]:.1f}s; {len(f)} available predictions", flush=True)
        seed_everything(seed)
        ckpt = f"emfn_{h}h.pt"
        info, _, p = train_and_evaluate_emfn(arrays["train"], arrays["validation"], arrays["test"],
            lookback_steps=L, horizon=h, epochs=cfg["train_epochs_emfn"], batch_size=cfg["batch_size"],
            device=device, capacity_mwh=capacity, save_model_path=str(run_dir/ckpt), selection_metric='loss')
        f = prediction_frame(raw["test"], p["input_indices"], h, p["y_true"], p["y_mean"],
            y_median=p["y_median"], p_pos=p["p_pos"], p_zero=p["p_zero"], alpha=p["alpha"], beta=p["beta"])
        record("EMFN (Proposed)", f, {k:info[k] for k in ["train_time_sec","best_epoch","stopped_epoch","best_val_loss","train_count","val_count"]}, ckpt)
        for name, factory in factories.items():
            seed_everything(seed)
            model = factory()
            ckpt = f"{slug(name)}_{h}h.pt"
            t0 = time.perf_counter()
            epochs = cfg["train_epochs_transformer"] if "Transformer" in name else cfg["train_epochs_baseline"]
            info, yt, yp = train_and_evaluate_torch_model(model, arrays["train"], arrays["validation"], arrays["test"],
                lookback_steps=L, horizon=h, epochs=epochs, batch_size=cfg["batch_size"], lr=.002,
                rated_capacity_mwh=capacity, device=device, save_model_path=str(run_dir/ckpt), selection_metric='loss')
            info["train_time_sec"] = time.perf_counter() - t0
            ds = WindTimeSeriesDataset(arrays["test"], L, h, capacity)
            record(name, prediction_frame(raw["test"], ds.valid_indices, h, yt, yp),
                {k:info[k] for k in ["train_time_sec","best_epoch","stopped_epoch","best_val_loss","train_count","val_count"]}, ckpt)
        seed_everything(seed)
        lgb = SangmyeongLightGBMForecaster(horizons=[h], rated_capacity_mwh=capacity)
        lgb.feature_cols = feature_cols
        t0 = time.perf_counter()
        booster = lgb.train_horizon(tab_splits["train"], tab_splits["validation"], h)
        elapsed = time.perf_counter()-t0
        _, yt, yp = lgb.evaluate(tab_splits["test"], h)
        ckpt = f"lightgbm_{h}h.txt"
        booster.save_model(str(run_dir/ckpt))
        record("LightGBM (+Weather)", prediction_frame(raw["test"], lgb.last_eval_indices+1, h, yt, yp),
               {"train_time_sec":elapsed, "best_iteration":booster.best_iteration}, ckpt)
        series = raw["test"].generation_mwh
        target = series.shift(-h)
        for name, model in [("Naive Persistence",NaivePersistence()), ("24h Diurnal Persistence",DiurnalPersistence())]:
            yp = model.predict(series, h)
            valid = np.isfinite(target) & np.isfinite(yp)
            origins = np.flatnonzero(valid)
            record(name, prediction_frame(raw["test"], origins+1, h, target[valid], yp[valid]), {"train_time_sec":0.0})
        aligned = align_predictions(frames)
        for name, f in aligned.items():
            f.to_parquet(run_dir/f"{slug(name)}_{h}h_common.parquet", index=False)
            scores = score_predictions(f, capacity, cfg["crps_grid_points"])
            scores.update(model=name, horizon=f"+{h}h", horizon_hours=h, seed=seed, run_id=run_id,
                weather_used="Persistence" not in name, train_time_sec=timings[name],
                coverage_pct=100*len(f)/len(raw["test"]), available_count=len(frames[name]))
            rows.append(scores)
            coverage.append(dict(model=name,horizon_hours=h,total_test_hours=len(raw["test"]),
                available_count=len(frames[name]),common_count=len(f),excluded_from_common=len(raw["test"])-len(f)))
        print(f"  COMMON +{h}h: {len(next(iter(aligned.values())))} target timestamps", flush=True)
        pd.DataFrame(rows).to_csv(run_dir/"benchmark.csv", index=False)
    pd.DataFrame(coverage).to_csv(run_dir/"coverage.csv",index=False)
    manifest["status"] = "complete"
    manifest["completed_at_kst"] = datetime.now(timezone(timedelta(hours=9))).isoformat()
    save_manifest()
    if set(horizons) == set(config["modeling"]["forecast_horizons"].values()):
        publish_run(run_dir)
    print(f"COMPLETE {run_dir}", flush=True)
    return run_dir


def publish_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete": raise ValueError("Only completed runs can be published")
    df = pd.read_csv(run_dir/"benchmark.csv")
    expected_horizons = {1, 3, 6, 9, 12, 24}
    if set(manifest["horizons"]) != expected_horizons or len(df) != 60:
        raise ValueError("Publishing the canonical tables requires all six horizons and ten models")
    if df.duplicated(["model", "horizon_hours"]).any() or not df.groupby("horizon_hours").size().eq(10).all():
        raise ValueError("Duplicate or missing model/horizon results")
    tables = ROOT/"reports/tables"
    for name in ["comprehensive_extended_metric_matrix", "full_multi_horizon_benchmark_matrix"]:
        df.to_csv(tables/f"{name}.csv",index=False)
    df[df.horizon_hours<=12].to_csv(tables/"intraday_dispatch_spectrum_benchmark.csv",index=False)
    shutil.copy2(run_dir/"coverage.csv",tables/"benchmark_coverage.csv")
    df[["model","horizon","count","bound_violation_pct","integrated_negative_mwh","min_pred_mwh","max_pred_mwh"]].to_csv(tables/"physical_boundary_violation_detailed.csv",index=False)
    for h in manifest["horizons"]:
        shutil.copy2(run_dir/f"emfn_{h}h.pt", ROOT/f"models/checkpoints/emfn_{h}h.pt")
        f = pd.read_parquet(run_dir/f"emfn__proposed_{h}h_predictions.parquet")
        np.savez_compressed(ROOT/f"models/checkpoints/emfn_{h}h_test_preds.npz", **{c:f[c].to_numpy() for c in f})
        (ROOT/f"models/checkpoints/emfn_{h}h.metadata.json").write_text(json.dumps({
            "run_id":manifest["run_id"],"horizon":h,"input_unit":"MWh","weather_scaler":manifest["weather_scaler"],
            "protocol":manifest["protocol"],"checkpoint_sha256":sha256(run_dir/f"emfn_{h}h.pt")},indent=2),encoding="utf-8")
    shutil.copy2(run_dir/"manifest.json", ROOT/"reports/corrected_run_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+")
    args = parser.parse_args()
    run_full_pipeline(args.horizons)
