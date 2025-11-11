"""
This module contains the models for the EMPROT project.
"""

from .transformer import (
    TemporalBackbone,
    ClassificationHead,
    ProteinTransformerClassificationOnly,
)
from .cta import CrossTemporalAttention

__all__ = [
    # Models
    'TemporalBackbone',
    'ClassificationHead',
    'ProteinTransformerClassificationOnly',
    
    # Core attention component
    'CrossTemporalAttention',
]
