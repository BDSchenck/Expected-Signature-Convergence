"""
Main source module for signature-based parameter estimation framework.
"""

# Core modules
from . import analytical
from . import numerical
# from . import calibration  # Commented out - using calibration_mega

__all__ = ['analytical', 'numerical']