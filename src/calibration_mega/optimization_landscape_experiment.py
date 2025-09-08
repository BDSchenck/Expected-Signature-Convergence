"""
Optimization Landscape Experiment Base Class (Stripped)

This module provides the base class for regime optimization experiments.
Only contains initialization needed by RegimeOptimizationExperiment.
All actual experiment logic is implemented in the child class.

Author: Bryson Schenck
Date: January 2025 (Stripped August 2025)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
import warnings
from scipy import stats
import logging

sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*degrees of freedom is <= 0.*')
warnings.filterwarnings('ignore', message='.*output with one or more elements was resized.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.autograd')

from src.numerical.paths import generate_ou_process
from src.numerical.parameters import generate_ou_parameters_for_regime
from src.calibration_mega.signatures import compute_parameter_mse, ExpectedSignatureCalibrator, RescaledSignatureCalibrator
from src.calibration_mega.mle_ultra_optimized import UltraOptimizedMLECalibrator
from src.calibration_mega.parameter_management import OptimizationParams
from src.calibration_mega.experiment_config import ExperimentConfig, ConfigPresets

# Phase 5 optimized imports
try:
    from src.calibration_mega.signatures import OptimizedRobustSignatureCalibrator
    from src.analytical.expected_signature_optimized import (
        compute_analytical_ou_expected_signature_cached,
        clear_generator_cache,
        get_cache_stats
    )
    OPTIMIZATIONS_AVAILABLE = True
    logger.info("Phase 5 optimizations loaded successfully")
except ImportError as e:
    OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Phase 5 optimizations not available: {e}")


class EnhancedOptimizationLandscapeExperiment:
    """
    Base class for regime optimization experiments.
    
    Only contains initialization - all methods are overridden in child class.
    This class sets up calibrators and parameters for the Enhancement Cascade:
    
    Phase 0: Analytical OU to identify optimal K*
    Phase 1: Enhanced MLE Refinement at optimal K with warm-start
    Phase 2: Signature Method Fine-Tuning with warm-start
    Phase 3: Method of Moments comprehensive baseline
    """
    
    def __init__(self, device=None, config: ExperimentConfig = None):
        """Initialize enhanced experiment configuration."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use provided config or default to optimized preset
        self.config = config if config is not None else ConfigPresets.optimized()
        
        # Experiment parameters
        self.d = 2  # Dimension
        self.M = 2  # Signature truncation level (optimal performance)
        self.T = 100.0  # Total path length
        self.n_steps = 262144  # Discretization steps (2^18, ~0.25M steps)
        self.n_monte_carlo = 1  # Number of Monte Carlo runs (back to 1 for testing)
        
        # Phase-specific K ranges (from user specifications)
        self.analytical_ou_K_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Phase 0 K values
        self.signature_K_values = [1024, 4096, 16384, 65536, 262144]  # Same as current signatures
        
        # Phase 0: Analytical OU doesn't need optimization parameters (closed-form solution)
        # But we keep a params object for interface compatibility
        self.analytical_ou_params = OptimizationParams(
            learning_rate=0.01,      # Not used - analytical solution
            learning_rate_end=0.01,  # Not used - analytical solution
            max_iterations=1,        # Not used - analytical solution
            patience=1,              # Not used - analytical solution
            gradient_clip_norm=1.0,
            weight_decay=1e-4
        )
        
        # Get parameters from centralized configuration
        self.enhanced_mle_params = self.config.get_mle_params()
        self.signature_params = self.config.get_signature_params()
        
        logger.info(f"Phase 0 Analytical OU params: Not applicable (closed-form solution)")
        logger.info(f"Phase 1 Enhanced MLE params: {self.enhanced_mle_params.log_parameters()}")
        logger.info(f"Phase 2 Signature params: {self.signature_params.log_parameters()}")
        
        # Parameter regime configuration
        self.regime_config = {
            'theta_eigenvalue_range': [0.05, 3.0],
            'mu_range': [-1.0, 1.0],
            'sigma_diag_range': [0.1, 2.0],
            'sigma_corr_level': [0.0, 1.0]
        }
        
        # Initialize calibrators - GPU-optimized without JIT compilation
        self.mle_calibrator = UltraOptimizedMLECalibrator(self.d, self.device, use_jit=False)
        
        # Use Phase 5 optimized signature calibrators if available
        if OPTIMIZATIONS_AVAILABLE:
            logger.info("Using Phase 5 optimized signature calibrators with caching and batching")
            self.expected_sig_calibrator = OptimizedRobustSignatureCalibrator(
                self.d, self.M, self.device, use_jit=False
            )
            self.rescaled_sig_calibrator = OptimizedRobustSignatureCalibrator(
                self.d, self.M, self.device, use_jit=False
            )
            self.use_optimized_signatures = True
        else:
            logger.info("Using standard signature calibrators")
            self.expected_sig_calibrator = ExpectedSignatureCalibrator(self.d, self.M, self.device)
            self.rescaled_sig_calibrator = RescaledSignatureCalibrator(self.d, self.M, self.device)
            self.use_optimized_signatures = False
        
        # Initialize classical benchmark calibrators
        logger.info("Initializing classical benchmark calibrators")
        from src.calibration_mega.analytical_ou_calibrator import AnalyticalOUCalibrator
        from src.calibration_mega.method_of_moments_calibrator import MethodOfMomentsCalibrator
        
        self.analytical_ou_calibrator = AnalyticalOUCalibrator(self.d, self.device)
        self.moments_calibrator = MethodOfMomentsCalibrator(self.d, self.device)
        
        logger.info(f"Enhanced experiment initialized on {self.device}")
    
    # All other methods are implemented in RegimeOptimizationExperiment