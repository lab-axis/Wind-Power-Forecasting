"""
EMFN Training, Validation, and Evaluation Module with Bernoulli-Beta Hurdle Head
- Target capacity: 21.0 MWh
- Target domain: [0, 21.0] MWh
- Early Stopping strictly on Validation NLL
- Dual Decoders: Mean for RMSE/MSE/R2, Median for MAE/WAPE
- Full Zero-Head Classification Metrics (AUROC, AUPRC, F1, Brier)
"""

import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.emfn import EMFN
from src.models.hurdle_beta import (
    BernoulliBetaHurdleLoss,
    decode_mixture_mean,
    decode_mixture_median,
    evaluate_zero_head,
)
from src.utils.metrics import evaluate_wind_forecast


class EMFNTimeSeriesDataset(Dataset):
    """
    Sliding window dataset for EMFN supporting target and exogenous sequences.
    """

    def __init__(
        self,
        target_series: np.ndarray,
        exog_matrix: Optional[np.ndarray] = None,
        lookback_len: int = 168,
        horizon: int = 1,
    ):
        self.target = target_series.astype(np.float32)
        self.exog = (
            exog_matrix.astype(np.float32) if exog_matrix is not None else None
        )
        self.lookback = lookback_len
        self.horizon = horizon

        self.valid_indices = []
        max_idx = len(self.target) - self.horizon
        for i in range(self.lookback, max_idx + 1):
            if not np.isnan(self.target[i + self.horizon - 1]):
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.valid_indices[idx]
        x_target = self.target[t - self.lookback : t]
        x_target = np.nan_to_num(x_target, nan=0.0)
        y = self.target[t + self.horizon - 1]

        t_target = torch.tensor(x_target, dtype=torch.float32).unsqueeze(-1)
        t_y = torch.tensor([y], dtype=torch.float32)

        if self.exog is not None:
            x_exog = self.exog[t - self.lookback : t]
            x_exog = np.nan_to_num(x_exog, nan=0.0)
            t_exog = torch.tensor(x_exog, dtype=torch.float32)
        else:
            t_exog = torch.empty(0, dtype=torch.float32)

        return t_target, t_exog, t_y


def train_emfn_model(
    model: EMFN,
    train_target: np.ndarray,
    val_target: np.ndarray,
    test_target: np.ndarray,
    train_exog: Optional[np.ndarray] = None,
    val_exog: Optional[np.ndarray] = None,
    test_exog: Optional[np.ndarray] = None,
    lookback_len: int = 168,
    horizon: int = 1,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.001,
    patience: int = 7,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trains EMFN with Bernoulli-Beta Hurdle NLL on Train, evaluates Early Stopping on Validation,
    and runs final frozen test evaluation on Test.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # 1. Datasets & Loaders
    train_ds = EMFNTimeSeriesDataset(
        train_target, train_exog, lookback_len=lookback_len, horizon=horizon
    )
    val_ds = EMFNTimeSeriesDataset(
        val_target, val_exog, lookback_len=lookback_len, horizon=horizon
    )
    test_ds = EMFNTimeSeriesDataset(
        test_target, test_exog, lookback_len=lookback_len, horizon=horizon
    )

    pin_mem = True if device == "cuda" else False
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem
    )

    # 2. Loss & Optimizer
    criterion = BernoulliBetaHurdleLoss(capacity_mwh=model.capacity_mwh, eps=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # 3. Training Loop
    history = {"train_nll": [], "val_nll": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_x_target, b_x_exog, b_y in train_loader:
            b_x_target = b_x_target.to(device)
            b_x_exog = b_x_exog.to(device) if b_x_exog.numel() > 0 else None
            b_y = b_y.to(device)

            optimizer.zero_grad()
            z_zero, z_alpha, z_beta = model(b_x_target, b_x_exog)
            loss = criterion(z_zero, z_alpha, z_beta, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x_target, b_x_exog, b_y in val_loader:
                b_x_target = b_x_target.to(device)
                b_x_exog = b_x_exog.to(device) if b_x_exog.numel() > 0 else None
                b_y = b_y.to(device)
                z_zero, z_alpha, z_beta = model(b_x_target, b_x_exog)
                val_loss += criterion(z_zero, z_alpha, z_beta, b_y).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["train_nll"].append(train_loss)
        history["val_nll"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 5. Load Best Validation Weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # 6. Test Set Prediction & Decoding
    model.eval()
    all_z_zero = []
    all_z_alpha = []
    all_z_beta = []
    all_trues = []

    with torch.no_grad():
        for b_x_target, b_x_exog, b_y in test_loader:
            b_x_target = b_x_target.to(device)
            b_x_exog = b_x_exog.to(device) if b_x_exog.numel() > 0 else None
            z_zero, z_alpha, z_beta = model(b_x_target, b_x_exog)

            all_z_zero.append(z_zero.cpu().numpy())
            all_z_alpha.append(z_alpha.cpu().numpy())
            all_z_beta.append(z_beta.cpu().numpy())
            all_trues.append(b_y.numpy())

    z_zero_cat = np.vstack(all_z_zero).flatten()
    z_alpha_cat = np.vstack(all_z_alpha).flatten()
    z_beta_cat = np.vstack(all_z_beta).flatten()
    y_true = np.vstack(all_trues).flatten()

    p_pos = 1.0 / (1.0 + np.exp(-z_zero_cat))
    alpha = np.log1p(np.exp(np.clip(z_alpha_cat, -20, 20))) + 1e-4
    beta = np.log1p(np.exp(np.clip(z_beta_cat, -20, 20))) + 1e-4

    # Dual point forecast decodings
    y_pred_mean = decode_mixture_mean(p_pos, alpha, beta, model.capacity_mwh)
    y_pred_median = decode_mixture_median(p_pos, alpha, beta, model.capacity_mwh)

    # 7. Metrics Calculation
    # Overall point forecast metrics
    metrics_mean = evaluate_wind_forecast(y_true, y_pred_mean, rated_capacity_mwh=model.capacity_mwh)
    metrics_median = evaluate_wind_forecast(y_true, y_pred_median, rated_capacity_mwh=model.capacity_mwh)

    # Zero Head classification metrics
    zero_metrics = evaluate_zero_head(y_true, p_pos)

    # Combined report dict (Canonical Expectation as standard benchmark prediction)
    mae_mean = float(metrics_mean["mae"])
    mae_med = float(metrics_median["mae"])
    med_improvement = ((mae_mean - mae_med) / max(mae_mean, 1e-6)) * 100.0

    result = {
        "model": "EMFN_BernoulliBeta_Hurdle",
        "horizon": f"+{horizon}h",
        "best_val_nll": float(best_val_loss),
        # Primary Canonical Benchmark Metrics (from single expectation vector E[Y|X])
        "mae_canonical": mae_mean,
        "nmae_canonical_pct": float(metrics_mean["nmae_pct"]),
        "rmse_canonical": float(metrics_mean["rmse"]),
        "nrmse_canonical_pct": float(metrics_mean["nrmse_pct"]),
        "r2_canonical": float(metrics_mean["r2"]),
        "corr_canonical": float(metrics_mean["corr"]),
        "wape_canonical_pct": float(metrics_mean["wape_pct"]),
        # Auxiliary Bayes Estimator (Median) for MAE-specific task
        "mae_median": mae_med,
        "nmae_median_pct": float(metrics_median["nmae_pct"]),
        "mae_median_improvement_pct": float(med_improvement),
        # Zero classification metrics
        "zero_auroc": zero_metrics["auroc"],
        "zero_auprc": zero_metrics["auprc_zero"],
        "zero_f1": zero_metrics["f1_zero"],
        "zero_brier": zero_metrics["brier_score_zero"],
        "zero_ece": zero_metrics["ece_zero"],
        # History
        "epochs_trained": len(history["train_nll"]),
        "y_true": y_true,
        "y_pred_mean": y_pred_mean,
        "y_pred_median": y_pred_median,
        "p_positive": p_pos,
    }

    return result
