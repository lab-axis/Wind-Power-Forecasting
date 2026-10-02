"""
EMFN v3 Backward Compatibility Redirect Module.
The canonical model file is now located at `src/models/emfn.py`.
"""

from src.models.emfn import (
    EMFN,
    EMFN_v3,
    CompositeHurdleLoss,
    CompositeHurdleLossV3,
    Chomp1d,
    TemporalBlock,
)

__all__ = [
    "EMFN",
    "EMFN_v3",
    "CompositeHurdleLoss",
    "CompositeHurdleLossV3",
    "Chomp1d",
    "TemporalBlock",
]
