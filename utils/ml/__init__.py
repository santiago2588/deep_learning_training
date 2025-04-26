"""
Machine learning utilities for the UoM Deep Learning Workshop.

This module provides functions for:
- Data scaling and normalization
- Model training and optimization
- Evaluation metrics and performance analysis
"""

__all__ = []

from .scaling import *
from .training import *
from .metrics import *

from .scaling import __all__ as scaling_all
from .training import __all__ as training_all
from .metrics import __all__ as metrics_all

__all__.extend(scaling_all)
__all__.extend(training_all)
__all__.extend(metrics_all)