"""
EMFN v2 Trainer and Evaluator Module
- Handles Multi-channel TCN + Weather Gating + Composite Hurdle Loss
- Validation-based Early Stopping (2024 Validation set)
- Evaluates 2025 Test Set on:
  1. Point Forecasts (Canonical Mean and Bayes Median)
  2. Hurdle Classification (AUROC, AUPRC, Brier, ECE)
  3. Physical Boundary Verification (0 <= Y <= 21 MWh)
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.history.emfn_v2 import EMFN_v2, CompositeHurdleLoss
from src.utils.metrics import evaluate_wind_forecast


class EMFNMultiChannelDataset(Dataset):
    """
    Sliding window dataset for EMFN v2:
    Outputs:
    - x_target: [L, 1] (in MWh)
    - x_weather: [L, C_w] (standardized weather features, optional)
    - y_true: [1] (in MWh)
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

        self.valid_indices = []
        max_idx = len(self.data) - self.horizon
        for i in range(self.lookback, max_idx + 1):
            if not np.isnan(self.data[i + self.horizon - 1, 0]):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x_raw = self.data[t - self.lookback : t, :]
        x_raw = np.nan_to_num(x_raw, nan=0.0)

        x_target = x_raw[:, :1]  # [L, 1]
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


def compute_expected_calibration_error(
    y_true_binary: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true_binary[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def train_and_evaluate_emfn_v2(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    lookback_steps: int = 24,
    horizon: int = 1,
    include_weather: bool = True,
    d_model: int = 64,
    dilations: Tuple[int, ...] = (1, 2, 4, 8),
    lambda_point: float = 1.0,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 6,
    capacity_mwh: float = 21.0,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Trains EMFN v2 and evaluates on 2025 Test Set.
    """
    n_weather = train_data.shape[1] - 1 if include_weather else 0

    train_ds = EMFNMultiChannelDataset(train_data, lookback_steps, horizon, include_weather)
    val_ds = EMFNMultiChannelDataset(val_data, lookback_steps, horizon, include_weather)
    test_ds = EMFNMultiChannelDataset(test_data, lookback_steps, horizon, include_weather)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EMFN_v2(
        lookback_len=lookback_steps,
        pred_len=1,
        n_weather_features=n_weather,
        d_model=d_model,
        dilations=dilations,
        capacity_mwh=capacity_mwh,
    ).to(device)

    criterion = CompositeHurdleLoss(
        capacity_mwh=capacity_mwh,
        lambda_point=lambda_point,
        huber_delta=0.05,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val_loss = float("inf")
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
        with torch.no_grad():
            for bx_target, bx_weather, by in val_loader:
                bx_target, by = bx_target.to(device), by.to(device)
                bx_weather = bx_weather.to(device) if include_weather else None
                z_zero, z_alpha, z_beta = model(bx_target, bx_weather)
                loss, _, _ = criterion(z_zero, z_alpha, z_beta, by)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_time = time.time() - t0

    # Load best weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Test Evaluation
    model.eval()
    all_y_true = []
    all_y_mean = []
    all_y_median = []
    all_p_pos = []

    with torch.no_grad():
        for bx_target, bx_weather, by in test_loader:
            bx_target = bx_target.to(device)
            bx_weather = bx_weather.to(device) if include_weather else None
            preds = model.predict_point_forecasts(bx_target, bx_weather)

            all_y_true.append(by.numpy().flatten())
            all_y_mean.append(preds["y_mean_rmse"].flatten())
            all_y_median.append(preds["y_median_mae"].flatten())
            all_p_pos.append(preds["p_positive"].flatten())

    y_true = np.concatenate(all_y_true)
    y_mean = np.concatenate(all_y_mean)
    y_median = np.concatenate(all_y_median)
    p_pos = np.concatenate(all_p_pos)

    # 1. Canonical Point Metrics (E[Y|X])
    metrics_canonical = evaluate_wind_forecast(y_true, y_mean, rated_capacity_mwh=capacity_mwh)

    # 2. Bayes Median Metrics
    metrics_median = evaluate_wind_forecast(y_true, y_median, rated_capacity_mwh=capacity_mwh)

    # 3. Hurdle Zero Classification Metrics
    # In data: 1 if positive (>0), 0 if zero
    is_positive_true = (y_true > 1e-4).astype(int)
    zero_auroc = float(roc_auc_score(is_positive_true, p_pos))
    # AUPRC for zero detection: predicting zero (1 - p_pos) against is_zero
    is_zero_true = 1 - is_positive_true
    p_zero = 1.0 - p_pos
    zero_auprc = float(average_precision_score(is_zero_true, p_zero))
    zero_brier = float(brier_score_loss(is_zero_true, p_zero))
    zero_ece = compute_expected_calibration_error(is_zero_true, p_zero)

    # 4. Physical Bound Violation Check
    neg_viol_rate = float(np.mean(y_mean < 0.0) * 100.0)
    cap_viol_rate = float(np.mean(y_mean > capacity_mwh) * 100.0)

    summary = {
        "model": "EMFN_v2 (+Weather)" if include_weather else "EMFN_v2 (Endogenous Only)",
        "horizon": f"+{horizon}h",
        "weather_used": include_weather,
        "mae_canonical": metrics_canonical["mae"],
        "rmse_canonical": metrics_canonical["rmse"],
        "nmae_canonical_pct": metrics_canonical["nmae_pct"],
        "r2_canonical": metrics_canonical["r2"],
        "corr_canonical": metrics_canonical["corr"],
        "wape_canonical_pct": metrics_canonical["wape_pct"],
        "mae_median": metrics_median["mae"],
        "median_gain_pct": (metrics_canonical["mae"] - metrics_median["mae"]) / metrics_canonical["mae"] * 100.0,
        "zero_auroc": zero_auroc,
        "zero_auprc": zero_auprc,
        "zero_brier": zero_brier,
        "zero_ece": zero_ece,
        "bound_violation_pct": neg_viol_rate + cap_viol_rate,
        "train_time_sec": train_time,
        "stopped_epoch": epoch + 1,
    }

    preds_dict = {
        "y_true": y_true,
        "y_mean": y_mean,
        "y_median": y_median,
        "p_pos": p_pos,
        "p_zero": p_zero,
    }

    return summary, y_mean, preds_dict
