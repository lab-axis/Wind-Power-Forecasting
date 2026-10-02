"""
EMFN v3 Trainer Backward Compatibility Redirect Module.
The canonical trainer file is now located at `src/models/emfn_trainer.py`.
"""

from src.models.emfn_trainer import (
    EMFNMultiChannelDataset,
    EMFNMultiChannelDatasetV3,
    compute_expected_calibration_error,
    train_and_evaluate_emfn,
    train_and_evaluate_emfn_v3,
)

__all__ = [
    "EMFNMultiChannelDataset",
    "EMFNMultiChannelDatasetV3",
    "compute_expected_calibration_error",
    "train_and_evaluate_emfn",
    "train_and_evaluate_emfn_v3",
]
