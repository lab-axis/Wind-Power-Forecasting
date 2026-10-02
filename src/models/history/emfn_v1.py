"""
EMFN: Exogenous Multi-scale Fusion Network with Bernoulli-Beta Hurdle Head
For bounded (0 <= Y <= 21 MWh) and zero-inflated (~17.5%) wind power generation forecasting.

Architecture:
1. Series Decomposition: Trend-Seasonal / Fluctuation separation.
2. Endogenous Multi-Scale Encoder: Dilated Causal Convolutions for target turbulence.
3. Exogenous Feature Encoder: Cross-scale projections for weather and temporal covariates.
4. Cross-Attention Fusion: Queries from target fluctuations, Keys/Values from exogenous variables.
5. Bernoulli-Beta Hurdle Head:
   - Output 1: z_zero (Logit for P(Y > 0 | X))
   - Output 2: z_alpha (Shape alpha for Beta distribution)
   - Output 3: z_beta (Shape beta for Beta distribution)
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.hurdle_beta import (
    BernoulliBetaHurdleLoss,
    decode_mixture_mean,
    decode_mixture_median,
)


class MovingAverageBlock(nn.Module):
    """
    Moving average block to separate trend and cyclical/fluctuation components.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        x_perm = x.permute(0, 2, 1)  # [B, C, L]
        # Reflection or replicated padding for edge stability
        front = x_perm[:, :, :1].repeat(1, 1, self.pad)
        end = x_perm[:, :, -1:].repeat(1, 1, self.pad)
        x_padded = torch.cat([front, x_perm, end], dim=-1)

        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        # Ensure exact length match
        trend = trend[:, :, : x.size(1)].permute(0, 2, 1)
        fluctuation = x - trend
        return trend, fluctuation


class DilatedConvBlock(nn.Module):
    """
    Dilated Causal 1D Convolution with Residual Connection and LayerNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        res = self.res_proj(x)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]  # Causal slicing
        out = self.act(out + res)
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.dropout(out)
        return out


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention layer fusing Endogenous Target representations with Exogenous Weather representations.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_val: torch.Tensor) -> torch.Tensor:
        # query: [B, L_q, D] (from endogenous target)
        # key_val: [B, L_kv, D] (from exogenous weather)
        attn_out, _ = self.mha(query, key_val, key_val)
        out = self.norm(query + self.dropout(attn_out))
        return out


class EMFN(nn.Module):
    """
    Exogenous Multi-scale Fusion Network with Bernoulli-Beta Hurdle Head.
    """

    def __init__(
        self,
        lookback_len: int = 168,
        pred_len: int = 1,
        n_exog_features: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        capacity_mwh: float = 21.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.n_exog_features = n_exog_features
        self.d_model = d_model
        self.capacity_mwh = capacity_mwh

        # 1. Target Decomposition
        self.decomp = MovingAverageBlock(kernel_size=25)

        # 2. Endogenous Fluctuation Stream (Multi-scale Dilated Convolutions)
        self.target_embed = nn.Linear(1, d_model)
        convs = []
        for d in dilations:
            convs.append(
                DilatedConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.target_convs = nn.ModuleList(convs)

        # 3. Exogenous Weather Stream (if exogenous features present)
        if n_exog_features > 0:
            self.exog_embed = nn.Linear(n_exog_features, d_model)
            self.exog_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Cross Attention
            self.cross_attn = CrossAttentionFusion(
                d_model=d_model, n_heads=n_heads, dropout=dropout
            )
        else:
            self.exog_embed = None
            self.cross_attn = None

        # 4. Trend & Autoregressive Projections (DLinear & AR persistence subsumption)
        self.trend_proj = nn.Linear(lookback_len, pred_len)
        self.ar_proj = nn.Linear(lookback_len, pred_len)

        # 5. Temporal Representation (Learned temporal projection)
        self.temporal_proj = nn.Linear(lookback_len, 1)

        # 6. Bernoulli-Beta Hurdle Output Heads
        latent_dim = d_model * 2 + pred_len * 2
        # Zero logit head (p = P(Y > 0 | X))
        self.head_zero = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # Beta alpha shape head (alpha > 0)
        self.head_alpha = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # Beta beta shape head (beta > 0)
        self.head_beta = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(
        self,
        x_target: torch.Tensor,
        x_exog: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_target: Past generation sequence [Batch, lookback_len, 1]
            x_exog: Past exogenous weather sequence [Batch, lookback_len, n_exog_features] (optional)

        Returns:
            z_zero: Logits for positive probability P(Y > 0 | X) [Batch, pred_len]
            z_alpha: Activation logits for Beta shape alpha [Batch, pred_len]
            z_beta: Activation logits for Beta shape beta [Batch, pred_len]
        """
        B, L, _ = x_target.shape

        # 1. Decompose Target into Trend and Fluctuation
        trend, fluc = self.decomp(x_target)

        # Linear trend extrapolation: [B, L, 1] -> [B, pred_len]
        trend_pred = self.trend_proj(trend.squeeze(-1))  # [B, pred_len]

        # 2. Endogenous Fluctuation Encoding (Multi-scale Dilated TCN)
        h_fluc = self.target_embed(fluc)  # [B, L, D]
        h_conv = h_fluc.permute(0, 2, 1)  # [B, D, L]
        for conv in self.target_convs:
            h_conv = conv(h_conv)
        h_target = h_conv.permute(0, 2, 1)  # [B, L, D]

        # 3. Exogenous Stream & Cross Attention
        if self.exog_embed is not None and x_exog is not None:
            h_exog = self.exog_embed(x_exog)  # [B, L, D]
            h_exog = self.exog_proj(h_exog)
            h_fused = self.cross_attn(query=h_target, key_val=h_exog)  # [B, L, D]
        else:
            h_fused = h_target

        # 4. Multi-scale Temporal Aggregation:
        # a) Last causal state (captures immediate high-frequency persistence)
        h_last = h_fused[:, -1, :]  # [B, D]
        # b) Learned temporal filter across all receptive lookback steps
        h_learned = self.temporal_proj(h_fused.permute(0, 2, 1)).squeeze(-1)  # [B, D]

        # 5. Direct Autoregressive Linear Shortcut (subsumes DLinear / AR baseline)
        ar_pred = self.ar_proj(x_target.squeeze(-1))  # [B, pred_len]

        # 6. Concatenate All Structural Latents
        latent = torch.cat([h_last, h_learned, trend_pred, ar_pred], dim=-1)  # [B, latent_dim]

        # 7. Bernoulli-Beta Heads
        z_zero = self.head_zero(latent)
        z_alpha = self.head_alpha(latent)
        z_beta = self.head_beta(latent)

        return z_zero, z_alpha, z_beta

    def predict_point_forecasts(
        self,
        x_target: torch.Tensor,
        x_exog: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computes both Mixture Mean (for RMSE/MSE/R2) and Mixture Median (for MAE).
        """
        self.eval()
        with torch.no_grad():
            z_zero, z_alpha, z_beta = self.forward(x_target, x_exog)
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
