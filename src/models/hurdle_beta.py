"""
Bernoulli-Beta Hurdle Model Module for Bounded and Zero-Inflated Wind Generation
- Target capacity: 21.0 MWh
- Target domain: y in [0, 21.0]
- Zero ratio: ~17.5%

Mathematical Formulation:
1. Zero probability head:
   p = P(Y > 0 | X) = sigmoid(z_zero)
   P(Y = 0 | X) = 1 - p

2. Positive branch (0 < y <= 21.0):
   x = y / 21.0,  clamped to [eps, 1 - eps]
   alpha = softplus(z_alpha) + eps
   beta  = softplus(z_beta)  + eps
   (Note: We do NOT force alpha, beta > 1 to preserve capacity for U-shaped / J-shaped edge densities)

3. Log-Likelihood:
   For y = 0:
     log P(Y = 0 | X) = log(1 - p) = -softplus(z_zero)
   For y > 0:
     log P(Y = y | X) = log(p) + log_beta_pdf(x; alpha, beta)
     where log_beta_pdf(x; alpha, beta) = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
                                         + (alpha - 1)*log(x) + (beta - 1)*log(1 - x)

4. Point Forecast Decodings:
   - For RMSE / MSE / R^2: Mixture conditional expectation:
       y_hat_RMSE = E[Y | X] = p * 21.0 * (alpha / (alpha + beta))
   - For MAE: Mixture conditional median:
       If (1 - p) >= 0.5:
           y_hat_MAE = 0.0
       Else:
           q_star = (p - 0.5) / p = 1.0 - (0.5 / p)
           y_hat_MAE = 21.0 * Beta_Quantile(q_star; alpha, beta)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)


class BernoulliBetaHurdleLoss(nn.Module):
    """
    Numerically stable Negative Log-Likelihood for Bernoulli-Beta Hurdle Model.
    """

    def __init__(self, capacity_mwh: float = 21.0, eps: float = 1e-4):
        super().__init__()
        self.capacity_mwh = capacity_mwh
        self.eps = eps

    def forward(
        self,
        z_zero: torch.Tensor,
        z_alpha: torch.Tensor,
        z_beta: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_zero: Logits for positive probability p = P(Y > 0 | X) [batch_size, ...]
            z_alpha: Unconstrained activations for alpha parameter
            z_beta:  Unconstrained activations for beta parameter
            y_true:  Observed generation in MWh [batch_size, ...] (y >= 0)

        Returns:
            nll: Mean negative log-likelihood scalar tensor
        """
        # 1. Parameter activations
        alpha = F.softplus(z_alpha) + self.eps
        beta = F.softplus(z_beta) + self.eps

        # 2. Binary mask for zero and positive observations
        is_positive = (y_true > 0).float()
        is_zero = 1.0 - is_positive

        # 3. Log-probabilities for the Hurdle classification
        # log(p) = log(sigmoid(z)) = -softplus(-z)
        # log(1 - p) = log(1 - sigmoid(z)) = -softplus(z)
        log_p = -F.softplus(-z_zero)
        log_1_minus_p = -F.softplus(z_zero)

        # 4. Normalized target for Beta distribution: x in [eps, 1 - eps]
        x_norm = y_true / self.capacity_mwh
        x_clamped = torch.clamp(x_norm, self.eps, 1.0 - self.eps)

        # 5. Beta Log-Likelihood via torch.lgamma and safe logs
        # log B(alpha, beta) = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
        log_b = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_beta_pdf = (
            -log_b
            + (alpha - 1.0) * torch.log(x_clamped)
            + (beta - 1.0) * torch.log(1.0 - x_clamped)
        )

        # 6. Combined Log-Likelihood
        # LL = is_zero * log(1 - p) + is_positive * (log(p) + log_beta_pdf)
        ll = is_zero * log_1_minus_p + is_positive * (log_p + log_beta_pdf)

        return -torch.mean(ll)


def decode_mixture_mean(
    p_pos: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    capacity_mwh: float = 21.0,
) -> np.ndarray:
    """
    Decodes point forecasts for RMSE/MSE/R2 optimization (Mixture Conditional Mean).
    E[Y | X] = p * capacity * (alpha / (alpha + beta))
    """
    p_pos = np.clip(p_pos, 0.0, 1.0)
    alpha = np.maximum(alpha, 1e-6)
    beta = np.maximum(beta, 1e-6)
    beta_mean = alpha / (alpha + beta)
    return p_pos * capacity_mwh * beta_mean


def decode_mixture_median(
    p_pos: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    capacity_mwh: float = 21.0,
) -> np.ndarray:
    """
    Decodes point forecasts for MAE optimization (Mixture Conditional Median).
    - If P(Y = 0 | X) = 1 - p >= 0.5 (i.e. p <= 0.5): Median is 0.0
    - If p > 0.5: Median solves F_Y(y) = (1 - p) + p * F_Beta(y/21) = 0.5
                  => F_Beta(y/21) = (p - 0.5) / p = 1 - 0.5 / p
                  => y_median = 21.0 * Q_Beta(1 - 0.5 / p; alpha, beta)
    """
    p_pos = np.clip(p_pos, 0.0, 1.0)
    alpha = np.maximum(alpha, 1e-6)
    beta = np.maximum(beta, 1e-6)

    medians = np.zeros_like(p_pos, dtype=np.float64)
    pos_mask = p_pos > 0.5

    if np.any(pos_mask):
        p_sub = p_pos[pos_mask]
        a_sub = alpha[pos_mask]
        b_sub = beta[pos_mask]

        q_star = 1.0 - (0.5 / p_sub)
        q_star = np.clip(q_star, 1e-6, 1.0 - 1e-6)

        # Scipy regularized incomplete beta inverse
        beta_quantiles = scipy_beta.ppf(q_star, a_sub, b_sub)
        medians[pos_mask] = capacity_mwh * beta_quantiles

    return np.clip(medians, 0.0, capacity_mwh)


def compute_ece(
    y_true_binary: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary probabilistic predictions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs <= bin_upper if i == n_bins - 1 else probs < bin_upper)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true_binary[mask])
            bin_conf = np.mean(probs[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_zero_head(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the Hurdle Zero-Classification Head.

    Args:
        y_true: Ground truth generation (MWh). 0 indicates zero-generation.
        p_pos: Model predicted probability that Y > 0 (P(Y > 0 | X)).
        threshold: Decision boundary for positive prediction (default: 0.5).

    Returns:
        Dictionary of classification metrics:
        - auroc: Area under ROC curve
        - auprc_positive: Average Precision for positive generation
        - auprc_zero: Average Precision for zero-generation (minority class 17.5%)
        - f1_zero: F1-score for zero detection
        - brier_score: Brier Score for probability calibration
        - ece_zero: Expected Calibration Error for zero probability
    """
    true_positive = (y_true > 0).astype(int)
    true_zero = (y_true == 0).astype(int)
    p_pos = np.clip(p_pos, 1e-7, 1.0 - 1e-7)
    p_zero = 1.0 - p_pos

    auroc = roc_auc_score(true_positive, p_pos)
    auprc_pos = average_precision_score(true_positive, p_pos)
    auprc_zero = average_precision_score(true_zero, p_zero)

    pred_zero = (p_zero >= (1.0 - threshold)).astype(int)
    f1_z = f1_score(true_zero, pred_zero, zero_division=0)
    brier = brier_score_loss(true_zero, p_zero)
    ece_z = compute_ece(true_zero, p_zero, n_bins=10)

    return {
        "auroc": float(auroc),
        "auprc_positive": float(auprc_pos),
        "auprc_zero": float(auprc_zero),
        "f1_zero": float(f1_z),
        "brier_score_zero": float(brier),
        "ece_zero": float(ece_z),
    }
