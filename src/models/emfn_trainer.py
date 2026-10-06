"""
EMFN Trainer and Evaluator Module
(Official Proposed Model for Sangmyeong Wind Farm Generation Forecasting)

- Handles Task-Decoupled Dual-Route EMFN Architecture
- Validation-based Early Stopping (2024 Validation set)
- Evaluates 2025 Test Set (8,760 hours) on:
  1. Point Forecasts (Canonical Mean and Bayes Median)
  2. Hurdle Classification (AUROC, AUPRC, Brier, ECE)
  3. Physical Boundary Verification (0 <= Y <= 21 MWh)
"""

import os
import sys
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.emfn import EMFN, CompositeHurdleLoss
from src.utils.metrics import evaluate_wind_forecast
from src.data.forecast_protocol import valid_window_indices
from src.models.hurdle_beta import compute_ece, evaluate_zero_head
from src.utils.checkpoint_selection import validation_crps, save_selection_history


class EMFNMultiChannelDataset(Dataset):
    """
    Sliding window dataset for EMFN.
    """
    def __init__(
        self,
        data_arr: np.ndarray,
        lookback_steps: int = 24,
        horizon: int = 1,
        include_weather: bool = True,
    ):
        self.data = np.array(data_arr, dtype=np.float32).copy()
        self.lookback = lookback_steps
        self.horizon = horizon
        self.include_weather = include_weather and (self.data.shape[1] > 1)

        checked = self.data if self.include_weather else self.data[:, :1]
        self.valid_indices = valid_window_indices(checked, self.lookback, self.horizon)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x_raw = self.data[t - self.lookback : t, :]

        x_target = x_raw[:, :1]  # [L, 1] in MWh
        if self.include_weather:
            x_weather = x_raw[:, 1:]  # [L, C_w]
        else:
            x_weather = np.zeros((self.lookback, 0), dtype=np.float32)

        y = self.data[t + self.horizon - 1, 0]  # [1] in MWh
        return (
            torch.tensor(x_target, dtype=torch.float32),
            torch.tensor(x_weather, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


# Backward compatibility alias
EMFNMultiChannelDatasetV3 = EMFNMultiChannelDataset


def compute_expected_calibration_error(
    y_true_binary: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compatibility wrapper with argument validation in the canonical ECE function."""
    return compute_ece(y_true_binary, probs, n_bins)


def train_and_evaluate_emfn(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    lookback_steps: int = 24,
    horizon: int = 1,
    include_weather: bool = True,
    d_model: int = 64,
    dilations: Tuple[int, ...] = (1, 2, 4, 8),
    lambda_point: float = 2.0,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 35,
    patience: int = 7,
    capacity_mwh: float = 21.0,
    device: str = "cuda",
    use_hf_skips: bool = True,
    use_selective_gate: bool = True,
    regression_mode: bool = False,
    model_name: Optional[str] = None,
    save_model_path: Optional[str] = None,
    selection_metric: str = "crps",
    crps_grid_points: int = 1001,
    model_factory=None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Trains EMFN and evaluates on 2025 Test Set.
    Supports comprehensive ablation matrix configurations.
    """
    n_weather = train_data.shape[1] - 1 if include_weather else 0
    if selection_metric not in ("crps", "loss"):
        raise ValueError("selection_metric must be crps or loss")

    train_ds = EMFNMultiChannelDataset(train_data, lookback_steps, horizon, include_weather)
    val_ds = EMFNMultiChannelDataset(val_data, lookback_steps, horizon, include_weather)
    test_ds = EMFNMultiChannelDataset(test_data, lookback_steps, horizon, include_weather)

    if any(len(ds) == 0 for ds in (train_ds, val_ds, test_ds)):
        raise ValueError("No finite windows in one or more splits")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Experimental constructors share this training/selection protocol; default
    # behavior and checkpoint layout remain those of the main EMFN model.
    model = (model_factory or EMFN)(
        lookback_len=lookback_steps,
        pred_len=1,
        n_weather_features=n_weather,
        d_model=d_model,
        dilations=dilations,
        capacity_mwh=capacity_mwh,
        use_hf_skips=use_hf_skips,
        use_selective_gate=use_selective_gate,
        regression_mode=regression_mode,
    ).to(device)

    criterion = CompositeHurdleLoss(
        capacity_mwh=capacity_mwh,
        lambda_point=lambda_point,
        huber_delta=0.05,
    )
    criterion_reg = nn.HuberLoss(delta=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_selection = float("inf")
    history = []
    best_weights = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx_target, bx_weather, by in train_loader:
            bx_target, by = bx_target.to(device), by.to(device)
            bx_weather = bx_weather.to(device) if include_weather else None

            optimizer.zero_grad()
            if regression_mode:
                y_pred_mwh, _, _ = model(bx_target, bx_weather)
                loss = criterion_reg(y_pred_mwh / capacity_mwh, by / capacity_mwh)
            else:
                z_zero, z_alpha, z_beta = model(bx_target, bx_weather)
                loss, _, _ = criterion(z_zero, z_alpha, z_beta, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions, val_p, val_a, val_b = [], [], [], []
        with torch.no_grad():
            for bx_target, bx_weather, by in val_loader:
                bx_target, by = bx_target.to(device), by.to(device)
                bx_weather = bx_weather.to(device) if include_weather else None
                if regression_mode:
                    y_pred_mwh, _, _ = model(bx_target, bx_weather)
                    loss = criterion_reg(y_pred_mwh / capacity_mwh, by / capacity_mwh)
                    val_predictions.append(y_pred_mwh.cpu().numpy().ravel())
                else:
                    z_zero, z_alpha, z_beta = model(bx_target, bx_weather)
                    loss, _, _ = criterion(z_zero, z_alpha, z_beta, by)
                    p=torch.sigmoid(z_zero)
                    a=torch.nn.functional.softplus(z_alpha)+1e-4
                    b=torch.nn.functional.softplus(z_beta)+1e-4
                    val_predictions.append((capacity_mwh*p*a/(a+b)).cpu().numpy().ravel())
                    val_p.append(p.cpu().numpy().ravel())
                    val_a.append(a.cpu().numpy().ravel())
                    val_b.append(b.cpu().numpy().ravel())
                val_loss += loss.item() * len(by)

        val_loss /= len(val_ds)
        val_targets=np.asarray(val_data)[np.asarray(val_ds.valid_indices)+horizon-1,0]
        params={} if regression_mode else dict(p_pos=np.concatenate(val_p),alpha=np.concatenate(val_a),beta=np.concatenate(val_b))
        val_crps=validation_crps(val_targets,np.concatenate(val_predictions),capacity_mwh,crps_grid_points,**params)
        selection=val_crps if selection_metric=="crps" else val_loss
        if not np.isfinite(selection): raise RuntimeError("Nonfinite validation selection score")
        scheduler.step(selection)
        history.append(dict(epoch=epoch+1,val_loss=val_loss,val_crps=val_crps,selection_score=selection))

        if selection < best_selection:
            best_selection=selection
            best_val_crps=val_crps
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_time = time.time() - t0

    if best_weights is None:
        raise RuntimeError("No finite validation checkpoint; training failed")
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        save_selection_history(save_model_path,history,selection_metric)

    # Test Evaluation
    model.eval()
    all_y_true = []
    all_y_mean = []
    all_y_median = []
    all_p_pos = []
    all_alpha, all_beta = [], []

    with torch.no_grad():
        for bx_target, bx_weather, by in test_loader:
            bx_target = bx_target.to(device)
            bx_weather = bx_weather.to(device) if include_weather else None
            preds = model.predict_point_forecasts(bx_target, bx_weather)

            all_y_true.append(by.numpy().flatten())
            all_y_mean.append(preds["y_mean_rmse"].flatten())
            all_y_median.append(preds["y_median_mae"].flatten())
            all_p_pos.append(preds["p_positive"].flatten())
            all_alpha.append(preds["alpha"].flatten())
            all_beta.append(preds["beta"].flatten())

    y_true = np.concatenate(all_y_true)
    y_mean = np.concatenate(all_y_mean)
    y_median = np.concatenate(all_y_median)
    p_pos = np.concatenate(all_p_pos)

    # 1. Canonical Point Metrics (E[Y|X])
    metrics_canonical = evaluate_wind_forecast(y_true, y_mean, rated_capacity_mwh=capacity_mwh)

    # 2. Bayes Median Metrics
    metrics_median = evaluate_wind_forecast(y_true, y_median, rated_capacity_mwh=capacity_mwh)

    # 3. Hurdle Zero Classification Metrics
    if regression_mode:
        zero_auroc = np.nan
        zero_auprc = np.nan
        zero_brier = np.nan
        zero_ece = np.nan
    else:
        zero = evaluate_zero_head(y_true, p_pos)
        zero_auroc = zero["auroc"]
        zero_auprc = zero["auprc_zero"]
        zero_brier = zero["brier_score_zero"]
        zero_ece = zero["ece_zero"]

    # 4. Physical Bounds Violation Check
    neg_viol_count = int(np.sum(y_mean < 0.0))
    cap_viol_count = int(np.sum(y_mean > capacity_mwh))
    neg_viol_rate = float(np.mean(y_mean < 0.0) * 100.0)
    cap_viol_rate = float(np.mean(y_mean > capacity_mwh) * 100.0)
    min_pred_val = float(np.min(y_mean))
    max_pred_val = float(np.max(y_mean))

    if model_name is None:
        if regression_mode:
            model_name = "EMFN (Deterministic Regression)"
        elif not include_weather:
            model_name = "EMFN (Endogenous Only)"
        elif not use_hf_skips:
            model_name = "EMFN (w/o HF Skips)"
        elif not use_selective_gate:
            model_name = "EMFN (w/o Selective Gate)"
        else:
            model_name = "EMFN (+Weather, Proposed)"

    summary = {
        "model": model_name,
        "horizon": f"+{horizon}h",
        "weather_used": include_weather,
        "use_hf_skips": use_hf_skips,
        "use_selective_gate": use_selective_gate,
        "regression_mode": regression_mode,
        "mae_canonical": metrics_canonical["mae"],
        "rmse_canonical": metrics_canonical["rmse"],
        "nmae_canonical_pct": metrics_canonical["nmae_pct"],
        "r2_canonical": metrics_canonical["r2"],
        "corr_canonical": metrics_canonical["corr"],
        "wape_canonical_pct": metrics_canonical["wape_pct"],
        "mae_median": metrics_median["mae"],
        "median_gain_pct": (metrics_canonical["mae"] - metrics_median["mae"]) / metrics_canonical["mae"] * 100.0 if not regression_mode else 0.0,
        "zero_auroc": zero_auroc,
        "zero_auprc": zero_auprc,
        "zero_brier": zero_brier,
        "zero_ece": zero_ece,
        "neg_viol_count": neg_viol_count,
        "neg_viol_rate_pct": neg_viol_rate,
        "cap_viol_count": cap_viol_count,
        "cap_viol_rate_pct": cap_viol_rate,
        "bound_violation_pct": neg_viol_rate + cap_viol_rate,
        "min_pred_mwh": min_pred_val,
        "max_pred_mwh": max_pred_val,
        "train_time_sec": train_time,
        "stopped_epoch": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "selection_metric": selection_metric,
        "best_val_crps": best_val_crps,
        "best_selection_score": best_selection,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
    }

    preds_dict = {
        "input_indices": np.asarray(test_ds.valid_indices),
        "alpha": np.concatenate(all_alpha),
        "beta": np.concatenate(all_beta),
        "y_true": y_true,
        "y_mean": y_mean,
        "y_median": y_median,
        "p_pos": p_pos if not regression_mode else np.ones_like(y_true),
        "p_zero": (1.0 - p_pos) if not regression_mode else np.zeros_like(y_true),
    }

    return summary, y_mean, preds_dict


# Backward compatibility alias
train_and_evaluate_emfn_v3 = train_and_evaluate_emfn

__all__ = [
    "EMFNMultiChannelDataset",
    "EMFNMultiChannelDatasetV3",
    "compute_expected_calibration_error",
    "train_and_evaluate_emfn",
    "train_and_evaluate_emfn_v3",
]
