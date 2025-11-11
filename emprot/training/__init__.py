"""Deprecated: legacy training utils namespace.

All evaluation helpers moved to `utils` package.
This module intentionally exports nothing to avoid new dependencies.
"""

from .trainer import EMPROTTrainer

__all__ = [
    'EMPROTTrainer',
]
