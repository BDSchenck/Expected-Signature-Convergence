"""
Calibration Module - Production Code for Parameter Optimization Agent
====================================================================

This module provides the essential calibration methods for the Enhancement Cascade
Architecture used in OU process parameter estimation experiments.

Essential Components for Agent:
- RegimeOptimizationExperiment: Main experiment framework for parameter optimization
- run_all_regime_experiments: Function to run all 4 regimes with combined analysis
- Enhancement Cascade: Analytical OU → Enhanced MLE → Signatures + Method of Moments

Author: Bryson Schenck
Date: August 2025 - Production Ready for Parameter Optimization
"""

# Core parameter management (agent modifies this)
from .parameter_management import OptimizationParams

# Enhancement cascade methods
from .analytical_ou_calibrator import AnalyticalOUCalibrator
from .method_of_moments_calibrator import MethodOfMomentsCalibrator
from .mle_ultra_optimized import UltraOptimizedMLECalibrator
from .signatures import (
    ExpectedSignatureCalibrator, 
    RescaledSignatureCalibrator
)

# Main experiment framework (what agent uses)
from .regime_optimization_experiment import (
    RegimeOptimizationExperiment,
    run_all_regime_experiments
)

# Utilities
from .signatures import compute_parameter_mse

# Public API (Production essentials only)
__all__ = [
    # Parameter system (agent modifies)
    'OptimizationParams',
    
    # Enhancement cascade methods
    'AnalyticalOUCalibrator',
    'MethodOfMomentsCalibrator', 
    'UltraOptimizedMLECalibrator',
    'ExpectedSignatureCalibrator', 
    'RescaledSignatureCalibrator',
    
    # Main experiments (agent uses these)
    'RegimeOptimizationExperiment',
    'run_all_regime_experiments',
    
    # Utilities
    'compute_parameter_mse'
]

__version__ = "3.0.0-production-agent-ready"