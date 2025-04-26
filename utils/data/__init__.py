"""
Data handling utilities for the UoM Deep Learning Workshop.

This module provides functions and utilities for:
1. Downloading datasets from various sources
2. Extracting and processing datasets
3. Working with specific datasets like UW-Madison GI Tract segmentation

Key components:
- download: Functions for downloading and extracting datasets
- uwmgi: Utilities for processing the UW-Madison GI Tract image segmentation dataset
"""

__all__ = []

# Import submodules and expose their public interfaces
from .download import *
from .download import __all__ as download_all
from .uwmgi import *
from .uwmgi import __all__ as uwmgi_all

# Update the module's __all__ to include submodules' public interfaces
__all__.extend(download_all)
__all__.extend(uwmgi_all)
