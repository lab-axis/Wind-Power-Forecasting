"""
EMFN v2: Multi-channel Temporal Convolutional Network with Weather State Gating
and Bernoulli-Beta Hurdle Head.

Key Innovations:
1. Multi-channel TCN Backbone: Direct cross-channel causal convolution of concatenated
   [Generation, Weather] sequences to capture instantaneous aerodynamic coupling (P ~ v^3)
   and temporal lag dynamics.
2. Auxiliary Weather State Gating: Extracts critical meteorological boundary conditions
   (e.g., low-wind cut-in, storm cut-out) to condition the Zero-Probability Head.
3. Bernoulli-Beta Hurdle Head: Guarantees strict physical bounds (0 <= Y <= 21 MWh) and
   explicitly models zero-generation (~17.5% prevalence).
4. Composite Supervised Hurdle Loss: Joint optimization of Hurdle NLL (probabilistic calibration)
   and Huber point forecast loss (direct MAE/RMSE minimization).
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
    """Causal slicing to guarantee strict temporal causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Dilated Causal 1D Convolution Block with Residual Skip & LayerNorm."""
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
        self.norm1 = nn.GroupNorm(1, n_outputs)  # Equivalent to LayerNorm across channels
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
    Composite Supervised Hurdle Loss:
    L_total = L_BCE(z_zero) + L_Beta(z_alpha, z_beta | y > 0) + lambda * Huber(E[Y|X], Y)
    """
    def __init__(
        self,
        capacity_mwh: float = 21.0,
        lambda_point: float = 1.0,
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
        # 1. Probabilistic Hurdle NLL (Classification + Beta density)
        loss_nll = self.hurdle_nll(z_zero, z_alpha, z_beta, y_true_mwh)

        # 2. Point Prediction Supervised Loss on Normalized Expected Value [0, 1]
        p_pos = torch.sigmoid(z_zero)
        alpha = F.softplus(z_alpha) + self.eps
        beta = F.softplus(z_beta) + self.eps
        y_mean_norm = p_pos * (alpha / (alpha + beta))
        y_true_norm = y_true_mwh / self.capacity_mwh

        loss_point = self.huber(y_mean_norm, y_true_norm)

        # 3. Composite Total Loss
        total_loss = loss_nll + self.lambda_point * loss_point
        return total_loss, loss_nll, loss_point


class EMFN_v2(nn.Module):
    """
    EMFN v2: Multi-channel Dilated TCN with Weather State Gating and Hurdle Beta Head.
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
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.n_weather_features = n_weather_features
        self.d_model = d_model
        self.capacity_mwh = capacity_mwh

        in_channels = 1 + n_weather_features  # Target + Weather channels

        # 1. Multi-channel TCN Backbone
        layers = []
        num_levels = len(dilations)
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

        # 2. Direct Autoregressive Linear Shortcut (subsumes AR / persistence)
        self.ar_shortcut = nn.Linear(lookback_len, pred_len)

        # 3. Weather State Gating Branch (Auxiliary path for boundary state detection)
        if n_weather_features > 0:
            # Summarizes recent weather (last 6 hours mean + latest instant)
            gate_in_dim = n_weather_features * 2
            self.weather_gate = nn.Sequential(
                nn.Linear(gate_in_dim, 32),
                nn.GELU(),
                nn.Linear(32, 16),
                nn.GELU(),
            )
            zero_gate_dim = 16
        else:
            self.weather_gate = None
            zero_gate_dim = 0

        # 4. Latent Fusion & Hurdle Heads
        # Latent dimensions:
        # h_last: d_model (64)
        # h_pool: d_model (64)
        # ar_pred: pred_len (1)
        latent_dim = d_model * 2 + pred_len

        # Zero Probability Head (Conditioned on latent + weather state)
        self.head_zero = nn.Sequential(
            nn.Linear(latent_dim + zero_gate_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )

        # Beta Alpha Head (Shape parameter > 0)
        self.head_alpha = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )

        # Beta Beta Head (Shape parameter > 0)
        self.head_beta = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )

    def forward(
        self,
        x_target: torch.Tensor,
        x_weather: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_target: Past generation sequence [Batch, lookback_len, 1] in MWh (or normalized)
            x_weather: Past weather sequence [Batch, lookback_len, n_weather_features] (optional)

        Returns:
            z_zero: Logits for positive probability P(Y > 0 | X) [Batch, pred_len]
            z_alpha: Activations for Beta alpha [Batch, pred_len]
            z_beta:  Activations for Beta beta [Batch, pred_len]
        """
        B, L, _ = x_target.shape

        # Normalize target generation into [0, 1] for network processing if needed
        # (Assuming x_target may be in [0, 21.0] MWh)
        if x_target.max() > 1.5:
            x_target_norm = x_target / self.capacity_mwh
        else:
            x_target_norm = x_target

        # 1. Multi-channel Input Fusion
        if self.n_weather_features > 0 and x_weather is not None:
            x_in = torch.cat([x_target_norm, x_weather], dim=-1)  # [B, L, 1 + n_weather]
        else:
            x_in = x_target_norm  # [B, L, 1]

        # 2. TCN Temporal Representation
        x_perm = x_in.transpose(1, 2)  # [B, C_in, L]
        h_seq = self.tcn(x_perm)       # [B, d_model, L]

        # 3. Multi-scale Pooling
        h_last = h_seq[:, :, -1]                                  # [B, d_model]
        h_pool = F.adaptive_avg_pool1d(h_seq, 1).squeeze(-1)      # [B, d_model]
        ar_pred = self.ar_shortcut(x_target_norm.squeeze(-1))     # [B, pred_len]

        latent = torch.cat([h_last, h_pool, ar_pred], dim=-1)     # [B, latent_dim]

        # 4. Weather State Gating
        if self.weather_gate is not None and x_weather is not None:
            w_window = min(6, L)
            w_recent = x_weather[:, -w_window:, :]
            w_mean = w_recent.mean(dim=1)
            w_last = x_weather[:, -1, :]
            w_feat = torch.cat([w_mean, w_last], dim=-1)
            w_state = self.weather_gate(w_feat)                   # [B, 16]
            zero_input = torch.cat([latent, w_state], dim=-1)
        else:
            zero_input = latent

        # 5. Output Heads
        z_zero = self.head_zero(zero_input)
        z_alpha = self.head_alpha(latent)
        z_beta = self.head_beta(latent)

        return z_zero, z_alpha, z_beta

    def predict_point_forecasts(
        self,
        x_target: torch.Tensor,
        x_weather: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computes both Mixture Mean (for RMSE/MSE/R2) and Mixture Median (for MAE).
        """
        self.eval()
        with torch.no_grad():
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
