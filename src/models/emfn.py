"""
EMFN: Exogenous Multiscale Fusion Network with Bernoulli-Beta Hurdle Head
(Official Proposed Model for Sangmyeong Wind Farm Generation Forecasting)

Architecture Highlights:
1. Shared Multi-channel Causal TCN Backbone:
   - Concurrently processes [Generation, Weather] with dilated causal convolutions.
2. Task-Decoupled Dual-Route Architecture:
   - Route A (Magnitude Branch):
     * Dedicated projection from shared spatio-temporal features.
     * High-frequency residual skip connections: immediate lag y_t, momentum (y_t - y_{t-1}),
       short-term variance, and direct autoregressive linear mapping.
     * Directly outputs Beta shape parameters (alpha, beta) for positive generation volume.
   - Route B (Zero-State Branch):
     * Dedicated projection from shared spatio-temporal features.
     * Domain-specific Selective Weather Gating: extracts cut-in wind speed proximity (v_t, v_mean, v_max)
       and historical zero streak dynamics.
     * Directly outputs Bernoulli logit z_zero for P(Y > 0 | X).
3. Eliminates Negative Transfer:
   - Prevents weather representations beneficial for magnitude prediction from perturbing
     the discrete boundary calibration of the zero classifier.
4. Composite Supervised Hurdle Objective:
   - Joint calibration of Hurdle NLL (probabilistic bounds) and weighted Huber point loss (direct MAE minimization).
5. Native Physical Boundary Guarantee:
   - Mathematical outputs strictly confined to [0, 21.0 MWh] with 0.00% bound violations.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hurdle_beta import (
    BernoulliBetaHurdleLoss,
    decode_mixture_mean,
    decode_mixture_median,
)


class Chomp1d(nn.Module):
    """Guarantees strict temporal causality by removing future padding."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Dilated Causal 1D Convolution with Residual Skip & GroupNorm."""
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.norm1 = nn.GroupNorm(1, n_outputs)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.norm2 = nn.GroupNorm(1, n_outputs)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(self.norm1(out))
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(self.norm2(out))
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class CompositeHurdleLoss(nn.Module):
    """
    Composite Hurdle Loss for EMFN with adaptive point loss scaling.
    L_total = L_BCE(z_zero) + L_Beta(alpha, beta | y > 0) + lambda * Huber(E[Y|X], Y)
    """
    def __init__(
        self,
        capacity_mwh: float = 21.0,
        lambda_point: float = 2.0,
        huber_delta: float = 0.05,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.capacity_mwh = capacity_mwh
        self.lambda_point = lambda_point
        self.eps = eps
        self.hurdle_nll = BernoulliBetaHurdleLoss(capacity_mwh=capacity_mwh, eps=eps)
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        z_zero: torch.Tensor,
        z_alpha: torch.Tensor,
        z_beta: torch.Tensor,
        y_true_mwh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Probabilistic Hurdle NLL
        loss_nll = self.hurdle_nll(z_zero, z_alpha, z_beta, y_true_mwh)

        # 2. Supervised Point Forecast Loss on Normalized Expected Value [0, 1]
        p_pos = torch.sigmoid(z_zero)
        alpha = F.softplus(z_alpha) + self.eps
        beta = F.softplus(z_beta) + self.eps
        y_mean_norm = p_pos * (alpha / (alpha + beta))
        y_true_norm = y_true_mwh / self.capacity_mwh

        loss_point = self.huber(y_mean_norm, y_true_norm)

        total_loss = loss_nll + self.lambda_point * loss_point
        return total_loss, loss_nll, loss_point


# Backward compatibility alias
CompositeHurdleLossV3 = CompositeHurdleLoss


class EMFN(nn.Module):
    """
    EMFN: Exogenous Multiscale Fusion Network (Proposed Main Architecture).
    Task-Decoupled Dual-Route Network with Multi-channel Causal TCN Backbone.

    Supports comprehensive ablation configurations:
    - use_hf_skips: enable/disable high-frequency momentum skips and AR shortcut
    - use_selective_gate: enable/disable domain-specific selective weather gating
    - regression_mode: enable standard deterministic regression instead of Hurdle Beta
    """
    def __init__(
        self,
        lookback_len: int = 24,
        pred_len: int = 1,
        n_weather_features: int = 6,
        d_model: int = 64,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        capacity_mwh: float = 21.0,
        dropout: float = 0.15,
        use_hf_skips: bool = True,
        use_selective_gate: bool = True,
        regression_mode: bool = False,
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.n_weather_features = n_weather_features
        self.d_model = d_model
        self.capacity_mwh = capacity_mwh
        self.use_hf_skips = use_hf_skips
        self.use_selective_gate = use_selective_gate
        self.regression_mode = regression_mode

        in_channels = 1 + n_weather_features

        # =====================================================================
        # 1. Shared Spatiotemporal Backbone: Multi-channel Causal TCN
        # =====================================================================
        layers = []
        for i, d in enumerate(dilations):
            in_ch = in_channels if i == 0 else d_model
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=d_model,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    padding=(kernel_size - 1) * d,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # =====================================================================
        # 2. Route A: Magnitude Branch (Positive Generation Volume)
        # =====================================================================
        self.mag_adapter = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_hf_skips:
            self.mag_hf_dim = 4
            self.ar_shortcut = nn.Linear(lookback_len, pred_len)
            mag_in_dim = d_model + self.mag_hf_dim + pred_len
        else:
            self.mag_hf_dim = 0
            self.ar_shortcut = None
            mag_in_dim = d_model

        if regression_mode:
            self.head_reg = nn.Sequential(
                nn.Linear(mag_in_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, pred_len),
            )
            self.head_alpha = None
            self.head_beta = None
        else:
            self.head_reg = None
            self.head_alpha = nn.Sequential(
                nn.Linear(mag_in_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, pred_len),
            )
            self.head_beta = nn.Sequential(
                nn.Linear(mag_in_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, pred_len),
            )

        # =====================================================================
        # 3. Route B: Zero-State Branch (Discrete Operational Status)
        # =====================================================================
        if not regression_mode:
            self.zero_adapter = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Selective Weather Gating (Physical cut-in / storm boundaries)
            if n_weather_features > 0 and use_selective_gate:
                self.gate_dim = 16
                self.selective_gate = nn.Sequential(
                    nn.Linear(8, 32),
                    nn.GELU(),
                    nn.Linear(32, self.gate_dim),
                    nn.GELU(),
                )
            else:
                self.gate_dim = 0
                self.selective_gate = None

            zero_in_dim = d_model + self.gate_dim + (pred_len if use_hf_skips else 0)
            self.head_zero = nn.Sequential(
                nn.Linear(zero_in_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, pred_len),
            )
        else:
            self.zero_adapter = None
            self.selective_gate = None
            self.head_zero = None

    def forward(
        self,
        x_target: torch.Tensor,
        x_weather: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x_target: Past generation sequence [B, L, 1] in MWh
            x_weather: Past weather sequence [B, L, C_w] (optional)
        """
        B, L, _ = x_target.shape

        # Normalize target generation into [0, 1] for network processing
        if x_target.max() > 1.5:
            x_target_norm = x_target / self.capacity_mwh
        else:
            x_target_norm = x_target

        # -------------------------------------------------------------
        # 1. Multi-channel Spatiotemporal Encoding
        # -------------------------------------------------------------
        if self.n_weather_features > 0 and x_weather is not None:
            x_in = torch.cat([x_target_norm, x_weather], dim=-1)  # [B, L, 1 + C_w]
        else:
            x_in = x_target_norm                                  # [B, L, 1]

        h_seq = self.tcn(x_in.transpose(1, 2))                    # [B, d_model, L]
        h_last = h_seq[:, :, -1]                                  # [B, d_model]
        h_pool = F.adaptive_avg_pool1d(h_seq, 1).squeeze(-1)      # [B, d_model]
        h_shared = torch.cat([h_last, h_pool], dim=-1)            # [B, d_model * 2]

        # -------------------------------------------------------------
        # 2. Route A: Magnitude Branch
        # -------------------------------------------------------------
        h_mag = self.mag_adapter(h_shared)                        # [B, d_model]

        if self.use_hf_skips:
            y_curr = x_target_norm[:, -1, :]                          # [B, 1]
            y_lag1 = x_target_norm[:, -2, :] if L >= 2 else y_curr    # [B, 1]
            y_lag2 = x_target_norm[:, -3, :] if L >= 3 else y_lag1    # [B, 1]
            w_6h = min(6, L)
            y_mean6 = x_target_norm[:, -w_6h:, :].mean(dim=1)         # [B, 1]
            hf_feats = torch.cat([y_curr, y_curr - y_lag1, y_curr - y_lag2, y_mean6], dim=-1)  # [B, 4]
            ar_pred = self.ar_shortcut(x_target_norm.squeeze(-1))     # [B, pred_len]
            mag_repr = torch.cat([h_mag, hf_feats, ar_pred], dim=-1)  # [B, mag_in_dim]
        else:
            mag_repr = h_mag
            ar_pred = None

        if self.regression_mode:
            y_norm_pred = self.head_reg(mag_repr)
            y_mwh_pred = y_norm_pred * self.capacity_mwh
            return y_mwh_pred, None, None

        z_alpha = self.head_alpha(mag_repr)
        z_beta = self.head_beta(mag_repr)

        # -------------------------------------------------------------
        # 3. Route B: Zero-State Branch
        # -------------------------------------------------------------
        h_zero = self.zero_adapter(h_shared)                      # [B, d_model]

        if self.selective_gate is not None and x_weather is not None:
            w_6h = min(6, L)
            ws = x_weather[:, :, 0:1]                             # wind speed channel [B, L, 1]
            press = x_weather[:, -1, 5:6]                         # pressure [B, 1]
            temp = x_weather[:, -1, 3:4]                          # temperature [B, 1]

            ws_last = ws[:, -1, :]                                # [B, 1]
            ws_mean6 = ws[:, -w_6h:, :].mean(dim=1)               # [B, 1]
            ws_max6 = ws[:, -w_6h:, :].max(dim=1).values          # [B, 1]

            is_zero_seq = (x_target_norm < 1e-4).float()
            is_zero_last = is_zero_seq[:, -1, :]                  # [B, 1]
            zero_freq6 = is_zero_seq[:, -w_6h:, :].mean(dim=1)    # [B, 1]
            zero_freq24 = is_zero_seq.mean(dim=1)                 # [B, 1]

            gate_inputs = torch.cat([
                ws_last, ws_mean6, ws_max6, press, temp,
                is_zero_last, zero_freq6, zero_freq24
            ], dim=-1)                                            # [B, 8]

            w_gate = self.selective_gate(gate_inputs)             # [B, 16]
            if self.use_hf_skips and ar_pred is not None:
                zero_repr = torch.cat([h_zero, w_gate, ar_pred], dim=-1)
            else:
                zero_repr = torch.cat([h_zero, w_gate], dim=-1)
        else:
            if self.use_hf_skips and ar_pred is not None:
                zero_repr = torch.cat([h_zero, ar_pred], dim=-1)
            else:
                zero_repr = h_zero

        z_zero = self.head_zero(zero_repr)
        return z_zero, z_alpha, z_beta

    def predict_point_forecasts(
        self,
        x_target: torch.Tensor,
        x_weather: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computes forecast outputs based on network mode (Hurdle vs Regression).
        """
        self.eval()
        with torch.no_grad():
            if self.regression_mode:
                y_pred_mwh, _, _ = self.forward(x_target, x_weather)
                y_pred_arr = y_pred_mwh.cpu().numpy()
                return {
                    "y_mean_rmse": y_pred_arr,
                    "y_median_mae": y_pred_arr,
                    "p_positive": np.ones_like(y_pred_arr),
                    "p_zero": np.zeros_like(y_pred_arr),
                    "alpha": np.zeros_like(y_pred_arr),
                    "beta": np.zeros_like(y_pred_arr),
                }

            z_zero, z_alpha, z_beta = self.forward(x_target, x_weather)
            p_pos = torch.sigmoid(z_zero).cpu().numpy()
            alpha = (F.softplus(z_alpha) + 1e-4).cpu().numpy()
            beta = (F.softplus(z_beta) + 1e-4).cpu().numpy()

        y_mean = decode_mixture_mean(p_pos, alpha, beta, self.capacity_mwh)
        y_median = decode_mixture_median(p_pos, alpha, beta, self.capacity_mwh)

        return {
            "y_mean_rmse": y_mean,
            "y_median_mae": y_median,
            "p_positive": p_pos,
            "p_zero": 1.0 - p_pos,
            "alpha": alpha,
            "beta": beta,
        }


# Backward compatibility alias
EMFN_v3 = EMFN

__all__ = ["EMFN", "EMFN_v3", "CompositeHurdleLoss", "CompositeHurdleLossV3"]
