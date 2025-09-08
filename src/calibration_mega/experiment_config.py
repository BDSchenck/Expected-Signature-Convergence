"""
Centralized experiment configuration for calibration experiments.
Single source of truth for hyperparameters with flexible learning rate control.
"""

from dataclasses import dataclass
from typing import Dict, Any
from .parameter_management import OptimizationParams


@dataclass
class ExperimentConfig:
    """
    Centralized configuration for all experiment parameters.
    Allows different learning rates for MLE vs Signature methods while keeping other parameters consistent.
    """
    # MLE Learning Rates
    mle_learning_rate: float = 0.05
    mle_learning_rate_end: float = 0.01
    
    # Signature Learning Rates  
    signature_learning_rate: float = 0.01
    signature_learning_rate_end: float = 0.005
    
    # Shared Parameters (consistent across all methods)
    patience: int = 100
    max_iterations: int = 1000
    tolerance: float = 1e-5
    convergence_factor: float = 0.995
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    
    def get_mle_params(self) -> OptimizationParams:
        """Get optimization parameters for Enhanced MLE."""
        return OptimizationParams(
            learning_rate=self.mle_learning_rate,
            learning_rate_end=self.mle_learning_rate_end,
            max_iterations=self.max_iterations,
            patience=self.patience,
            gradient_clip_norm=self.gradient_clip_norm,
            weight_decay=self.weight_decay,
            tolerance=self.tolerance,
            convergence_factor=self.convergence_factor
        )
    
    def get_signature_params(self) -> OptimizationParams:
        """Get optimization parameters for Signature methods."""
        return OptimizationParams(
            learning_rate=self.signature_learning_rate,
            learning_rate_end=self.signature_learning_rate_end,
            max_iterations=self.max_iterations,
            patience=self.patience,
            gradient_clip_norm=self.gradient_clip_norm,
            weight_decay=self.weight_decay,
            tolerance=self.tolerance,
            convergence_factor=self.convergence_factor
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mle_learning_rate": self.mle_learning_rate,
            "mle_learning_rate_end": self.mle_learning_rate_end,
            "signature_learning_rate": self.signature_learning_rate,
            "signature_learning_rate_end": self.signature_learning_rate_end,
            "patience": self.patience,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "convergence_factor": self.convergence_factor,
            "gradient_clip_norm": self.gradient_clip_norm,
            "weight_decay": self.weight_decay
        }
    
    def __str__(self) -> str:
        """String representation for logging."""
        return (f"ExperimentConfig("
                f"MLE: {self.mle_learning_rate}>{self.mle_learning_rate_end}, "
                f"Sig: {self.signature_learning_rate}>{self.signature_learning_rate_end}, "
                f"patience={self.patience})")


# Pre-defined configurations for common experiments
class ConfigPresets:
    """Pre-defined configurations for common experiment types."""
    
    @staticmethod
    def baseline() -> ExperimentConfig:
        """Original baseline configuration."""
        return ExperimentConfig(
            mle_learning_rate=0.01,
            mle_learning_rate_end=0.005,
            signature_learning_rate=0.01,
            signature_learning_rate_end=0.005,
            patience=100  # FIXED: Changed from 200 to 100 for fair comparison
        )
    
    @staticmethod
    def optimized() -> ExperimentConfig:
        """Optimized configuration based on learning rate experiments."""
        return ExperimentConfig(
            mle_learning_rate=0.05,
            mle_learning_rate_end=0.01,
            signature_learning_rate=0.05,  # Apply same optimized rates to signatures
            signature_learning_rate_end=0.01,
            patience=100
        )
    
    @staticmethod
    def mixed_optimal() -> ExperimentConfig:
        """Mixed configuration: optimized MLE, baseline signatures."""
        return ExperimentConfig(
            mle_learning_rate=0.05,         # Optimized for MLE
            mle_learning_rate_end=0.01,
            signature_learning_rate=0.01,   # Keep signatures at baseline
            signature_learning_rate_end=0.005,
            patience=100
        )
    
    @staticmethod
    def custom(mle_lr: float, mle_lr_end: float, 
               sig_lr: float, sig_lr_end: float, 
               patience: int = 100) -> ExperimentConfig:
        """Create custom configuration with specified learning rates."""
        return ExperimentConfig(
            mle_learning_rate=mle_lr,
            mle_learning_rate_end=mle_lr_end,
            signature_learning_rate=sig_lr,
            signature_learning_rate_end=sig_lr_end,
            patience=patience
        )