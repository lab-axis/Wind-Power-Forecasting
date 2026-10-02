"""
Sangmyeong Wind Farm Generation Forecasting - Models Package
- Proposed Main Model: EMFN (Exogenous Multiscale Fusion Network)
- Baseline Models: LightGBM, DLinear, CNN-LSTM, LSTM, TCN, Transformer, Persistence
"""

from src.models.emfn import EMFN, CompositeHurdleLoss
from src.models.emfn_trainer import train_and_evaluate_emfn, EMFNMultiChannelDataset
from src.models.lightgbm_model import SangmyeongLightGBMForecaster
from src.models.dlinear import DLinearForecaster
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.rnn import LSTMForecaster, GRUForecaster
from src.models.tcn_model import TCNForecaster
from src.models.transformer import TimeSeriesTransformerForecaster
from src.models.persistence import NaivePersistence, DiurnalPersistence
from src.models.torch_trainer import train_and_evaluate_torch_model
from src.models.hurdle_beta import (
    BernoulliBetaHurdleLoss,
    decode_mixture_mean,
    decode_mixture_median,
)

__all__ = [
    # Proposed Architecture
    "EMFN",
    "CompositeHurdleLoss",
    "train_and_evaluate_emfn",
    "EMFNMultiChannelDataset",
    # Machine Learning & Deep Learning Baselines
    "SangmyeongLightGBMForecaster",
    "DLinearForecaster",
    "CNNLSTMForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "TCNForecaster",
    "TimeSeriesTransformerForecaster",
    "NaivePersistence",
    "DiurnalPersistence",
    # Utilities & Losses
    "train_and_evaluate_torch_model",
    "BernoulliBetaHurdleLoss",
    "decode_mixture_mean",
    "decode_mixture_median",
]
