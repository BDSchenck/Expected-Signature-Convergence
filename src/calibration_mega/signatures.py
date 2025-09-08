"""
Consolidated Signature Methods for Calibration Mega
====================================================

This module consolidates all signature-related calibrators from the original 3 files:
- signature_robust.py -> RobustSignatureCalibrator (base implementation)
- signature_robust_optimized.py -> OptimizedRobustSignatureCalibrator (with caching)
- signature_calibrators_unified.py -> ExpectedSignatureCalibrator, RescaledSignatureCalibrator (wrappers)

The block_signature_calibration.py remains separate as it contains MLE utilities.

Author: Bryson Schenck (refactored)
Date: August 2025
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='.*output with one or more elements was resized.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.autograd')

# Import from src dependencies (these are allowed and expected)
from src.numerical.estimators import batched_calculate_signature_estimator
from src.analytical.expected_signature import (
    compute_analytical_ou_expected_signature,
    construct_generator_matrix
)

# Import from local calibration_mega files
from .block_signature_calibration import taylor_exp_action
from .parameter_management import (
    OptimizationParams, ParameterInjector, CalibrationResult, CalibratorInterface
)

# Import utilities from block_signature_calibration (kept separate)
from .block_signature_calibration import (
    taylor_exp_action,
    scaling_squaring_exp_action,
    BatchedMLECalibrator,
    compute_parameter_mse
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE ROBUST SIGNATURE CALIBRATOR
# =============================================================================

class RobustSignatureCalibrator(CalibratorInterface):
    """
    Robust signature calibrator with enhanced convergence for all T values.
    
    This calibrator uses the following unified parameters:
    1. Same learning rate as MLE (0.01)
    2. Same patience as MLE (10)  
    3. Better initialization from empirical moments
    4. Same gradient clipping as MLE (1.0)
    5. Smart parameter projection (same as MLE)
    """
    
    def __init__(self, d, M, device, use_jit=True):
        super().__init__(device)
        self.d = d
        self.M = M
        self.sig_dim = sum(d**i for i in range(M+1))
        self.use_jit = use_jit
        
        # JIT compile critical operations
        if use_jit and hasattr(torch, 'compile'):
            logger.info("JIT compiling critical operations for robust calibrator...")
            
            self._compute_eigenvalues_jit = torch.compile(
                self._compute_eigenvalues_core,
                mode="reduce-overhead"
            )
            
            self._compute_constraint_penalty_jit = torch.compile(
                self._compute_constraint_penalty_core, 
                mode="reduce-overhead"
            )
        else:
            self._compute_eigenvalues_jit = self._compute_eigenvalues_core
            self._compute_constraint_penalty_jit = self._compute_constraint_penalty_core
        
        logger.info(f"Robust signature calibrator initialized (JIT={'on' if use_jit else 'off'})")
    
    def _compute_eigenvalues_core(self, matrix):
        """Core eigenvalue computation - JIT compiled for speed."""
        return torch.linalg.eigvalsh(matrix)
    
    def _compute_constraint_penalty_core(self, theta, sigma):
        """
        Core constraint penalty computation with adaptive weighting.
        """
        penalty = 0.0
        
        # UNIFIED: Same constraint penalties as MLE (no advantages)
        try:
            theta_eigvals = torch.linalg.eigvalsh(theta)
            min_eig_theta = torch.min(theta_eigvals)
            penalty += 50.0 * torch.relu(0.01 - min_eig_theta)  # Same as MLE
        except:
            penalty += 1000.0  # Same as MLE
        
        # UNIFIED: Same sigma penalties as MLE
        try:
            sigma_eigvals = torch.linalg.eigvalsh(sigma)
            min_eig_sigma = torch.min(sigma_eigvals)
            penalty += 50.0 * torch.relu(1e-4 - min_eig_sigma)  # Same as MLE
        except:
            penalty += 1000.0  # Same as MLE
            
        return penalty
    
    def estimate(
        self,
        path: torch.Tensor,
        K: int,
        T: float,
        params: OptimizationParams,
        init_params=None,
        verbose: bool = False,
        method: str = "standard"
    ) -> CalibrationResult:
        """
        Main estimation method implementing CalibratorInterface.
        
        Args:
            path: Observed path
            K: Number of blocks
            T: Time horizon
            params: MANDATORY optimization parameters
            init_params: Optional initial parameters
            verbose: Logging flag
            method: "standard" or "rescaled"
            
        Returns:
            CalibrationResult with full parameter tracking
        """
        # Validate inputs
        self._validate_inputs(path, K, T, params)
        
        # Log configuration
        if verbose:
            self._log_configuration(f"RobustSignature-{method}", K, T, params)
        
        # Delegate to appropriate method
        if method == "standard":
            return self.estimate_standard_robust(path, K, T, params, init_params, verbose)
        elif method == "rescaled":
            return self.estimate_rescaled_robust(path, K, T, params, init_params, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _get_unified_params(self, params: OptimizationParams = None) -> OptimizationParams:
        """
        Get unified parameters, using defaults if not provided.
        
        Args:
            params: Optional optimization parameters
            
        Returns:
            OptimizationParams object
        """
        if params is None:
            # Use default unified parameters
            params = OptimizationParams(
                learning_rate=0.01,
                max_iterations=2000,
                patience=10,
                gradient_clip_norm=1.0,
                weight_decay=1e-4
            )
        return params
    
    def _compute_smart_initialization(self, path, K, T):
        """
        Compute smart initialization from empirical path statistics.
        Enhanced for high K scenarios where blocks are very short.
        """
        n_steps = path.shape[0]
        dt = T / (n_steps - 1)
        
        # Empirical mean (time-averaged)
        mu_emp = torch.mean(path, dim=0)
        
        # CRITICAL FIX: Better sigma estimation for high K
        # When K is large, blocks are short and increment-based estimation is noisy
        if K >= 4096:
            # Use full path variance scaled appropriately
            path_centered = path - mu_emp
            sigma_emp = torch.cov(path_centered.T)
            # Scale by mean reversion time estimate
            sigma_emp = sigma_emp * 2.0  # Approximate steady-state scaling
        else:
            # Standard increment-based estimation for lower K
            increments = (path[1:] - path[:-1]) / torch.sqrt(torch.tensor(dt, device=self.device))
            sigma_emp = torch.cov(increments.T)
        
        # Ensure positive definite with appropriate minimum eigenvalue
        eigvals, eigvecs = torch.linalg.eigh(sigma_emp)
        min_eig = 0.01 if K >= 4096 else 1e-4  # Higher minimum for high K stability
        eigvals = torch.clamp(eigvals, min=min_eig)
        sigma_emp = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        
        # CRITICAL FIX: Better theta initialization for high K
        # High K means short blocks where mean reversion is less observable
        var_x = torch.var(path, dim=0)
        if K >= 4096:
            # For high K, use conservative theta estimate
            theta_diag = torch.ones(self.d, device=self.device) * 1.0  # Moderate mean reversion
        elif n_steps > 100:
            # Standard autocorrelation-based estimation for lower K
            lag = min(50, n_steps // 10)
            autocorr = torch.mean((path[:-lag] - mu_emp) * (path[lag:] - mu_emp), dim=0) / (var_x + 1e-8)
            autocorr = torch.clamp(autocorr, min=0.01, max=0.99)
            theta_diag = -torch.log(autocorr) / (lag * dt)
            theta_diag = torch.clamp(theta_diag, min=0.1, max=5.0)
        else:
            theta_diag = torch.ones(self.d, device=self.device) * 0.5
        
        theta_emp = torch.diag(theta_diag)
        
        return theta_emp, mu_emp, sigma_emp
    
    def estimate_standard_robust(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Standard signature calibration with unified parameters.
        
        Args:
            path: Observed path
            K: Number of blocks
            T: Time horizon
            params: OptimizationParams (unified parameters)
            init_params: Optional initial parameters
            verbose: Logging flag
            
        Returns:
            CalibrationResult with full parameter tracking
        """
        # Get unified params
        params = self._get_unified_params(params)
        
        if verbose:
            logger.info(f"Robust standard signature calibration for K={K}, T={T}...")
            logger.info(f"Using parameters: {params.log_parameters()}")
            start_time = time.time()
        
        # Compute empirical signature
        paths_batch = path.unsqueeze(0)
        dummy_true = torch.zeros(1, self.sig_dim, device=self.device, dtype=torch.float64)
        
        empirical_sig = batched_calculate_signature_estimator(
            paths_batch, self.M, K, dummy_true, return_estimator=True
        ).squeeze(0)
        
        block_duration = T / K
        
        # Initialize with smart initialization if not provided
        if init_params is None:
            theta_init, mu_init, sigma_init = self._compute_smart_initialization(path, K, T)
        else:
            theta_init, mu_init, sigma_init = init_params
        
        # For T=1, we focus on convergence over accuracy
        max_restarts = 1  # No restarts - always report convergence based on patience
        best_overall_loss = float('inf')
        best_overall_params = None
        overall_converged = False
        
        for restart in range(max_restarts):
            if restart > 0:
                # Add noise for restart
                noise_scale = 0.1 * (restart + 1)
                theta_init = theta_init + noise_scale * torch.randn_like(theta_init)
                mu_init = mu_init + noise_scale * torch.randn_like(mu_init)
                sigma_init = sigma_init + noise_scale * torch.randn_like(sigma_init)
                
                # Ensure positive definite after noise
                eigvals, eigvecs = torch.linalg.eigh(theta_init)
                eigvals = torch.clamp(eigvals, min=0.1)
                theta_init = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                
                eigvals, eigvecs = torch.linalg.eigh(sigma_init)
                eigvals = torch.clamp(eigvals, min=1e-3)
                sigma_init = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            
            theta = theta_init.clone().requires_grad_(True)
            mu = mu_init.clone().requires_grad_(True)
            
            STABILITY_EPS = 1e-6
            L_sigma = torch.linalg.cholesky(
                sigma_init + STABILITY_EPS * torch.eye(self.d, device=self.device, dtype=torch.float64)
            )
            L_sigma.requires_grad_(True)
            
            # Create unified optimizer (same for all K values for fairness)
            optimizer = ParameterInjector.create_unified_optimizer(
                [theta, mu, L_sigma], params, self.device
            )
            
            # Create unified scheduler (constant LR for fairness)
            scheduler = ParameterInjector.create_unified_scheduler(optimizer, params)
            
            # Convergence tracking
            best_loss = float('inf')
            best_params = None
            no_improve = 0
            converged = False
            
            for iteration in range(params.max_iterations):
                optimizer.zero_grad()
                
                sigma_reconstruct = L_sigma @ L_sigma.T
                
                # Expected signature computation with error handling
                try:
                    expected_sig = compute_analytical_ou_expected_signature(
                        block_duration, self.d, self.M,
                        theta.unsqueeze(0),
                        mu.unsqueeze(0),
                        sigma_reconstruct.unsqueeze(0),
                        use_time_augmentation=False
                    ).squeeze(0)
                except Exception as e:
                    # Stabilize parameters (ensure correct dtype)
                    theta_stable = theta + 0.1 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                    sigma_stable = sigma_reconstruct + 1e-3 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                    expected_sig = compute_analytical_ou_expected_signature(
                        block_duration, self.d, self.M,
                        theta_stable.unsqueeze(0),
                        mu.unsqueeze(0),
                        sigma_stable.unsqueeze(0),
                        use_time_augmentation=False
                    ).squeeze(0)
                
                # Main loss with normalization for stability
                diff = empirical_sig - expected_sig
                loss = torch.mean(diff ** 2)  # Mean instead of sum for scale invariance
                
                # Adaptive constraint penalties
                constraint_penalty = self._compute_constraint_penalty_jit(theta, sigma_reconstruct)
                
                # Scale constraint penalty based on T
                if T <= 1.0:
                    constraint_weight = 0.1  # Lower weight for short paths
                else:
                    constraint_weight = 1.0
                
                total_loss = loss + constraint_weight * constraint_penalty
                
                total_loss.backward()
                
                # Track best parameters
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_params = (theta.detach().clone(), mu.detach().clone(), sigma_reconstruct.detach().clone())
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Check convergence with unified logic
                if ParameterInjector.validate_convergence_logic(
                    no_improve, params.patience, params
                ):
                    converged = True
                    break
                
                # Unified gradient clipping
                ParameterInjector.clip_gradients([theta, mu, L_sigma], params)
                
                optimizer.step()
                scheduler.step()
                
                # Smart periodic projection (like MLE) - only fix problematic parameters
                if iteration % 20 == 0:
                    with torch.no_grad():
                        # Quick check if projection is needed (fast diagonal check)
                        theta_diag = torch.diagonal(theta)
                        theta_needs_fix = torch.min(theta_diag) < 0.005
                        
                        if theta_needs_fix:
                            # Only project theta if actually problematic
                            try:
                                eigvals, eigvecs = torch.linalg.eigh(theta)
                                eigvals = torch.clamp(eigvals, min=0.05)
                                theta.data = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                            except:
                                theta.data = theta + 0.1 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                        
                        # Quick check for L_sigma (check diagonal of L matrix)
                        L_diag = torch.diagonal(L_sigma)
                        L_needs_fix = torch.min(L_diag) < 1e-5
                        
                        if L_needs_fix:
                            # Only project L_sigma if actually problematic
                            try:
                                sigma_temp = L_sigma @ L_sigma.T
                                eigvals_s, eigvecs_s = torch.linalg.eigh(sigma_temp)
                                eigvals_s = torch.clamp(eigvals_s, min=1e-3)
                                sigma_valid = eigvecs_s @ torch.diag(eigvals_s) @ eigvecs_s.T
                                L_sigma.data = torch.linalg.cholesky(
                                    sigma_valid + STABILITY_EPS * torch.eye(self.d, device=self.device)
                                )
                            except:
                                L_sigma.data = L_sigma + 1e-3 * torch.eye(self.d, device=self.device, dtype=torch.float64)
            
            # Check if this restart is better
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_params = best_params
                overall_converged = converged
                
                # If we converged with reasonable loss, stop restarts
                if converged and best_loss < 10.0:  # More realistic threshold
                    break
        
        # Final parameter handling
        if best_overall_params is None:
            sigma_final = L_sigma @ L_sigma.T
            best_overall_params = (theta.detach(), mu.detach(), sigma_final.detach())
        
        # Track optimization time
        elapsed = time.time() - start_time if verbose else 0.0
        if verbose:
            logger.info(f"Robust standard signature complete in {elapsed:.2f}s, converged={overall_converged}")
        
        # Return CalibrationResult with full tracking
        theta_final, mu_final, sigma_final = best_overall_params
        return CalibrationResult(
            theta=theta_final,
            mu=mu_final,
            sigma=sigma_final,
            converged=overall_converged,
            final_loss=best_overall_loss,
            iterations_used=iteration if 'iteration' in locals() else 0,
            parameters_used=params,
            method_name="Standard Signature",
            K_blocks=K,
            T_horizon=T,
            optimization_time=elapsed
        )
    
    def estimate_rescaled_robust(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Rescaled signature calibration with unified parameters.
        
        Args:
            path: Observed path
            K: Number of blocks
            T: Time horizon
            params: OptimizationParams (unified parameters)
            init_params: Optional initial parameters
            verbose: Logging flag
            
        Returns:
            CalibrationResult with full parameter tracking
        """
        # Get unified params
        params = self._get_unified_params(params)
        
        if verbose:
            logger.info(f"Robust rescaled signature calibration for K={K}, T={T}...")
            logger.info(f"Using parameters: {params.log_parameters()}")
            start_time = time.time()
        
        # Compute empirical signature
        paths_batch = path.unsqueeze(0)
        dummy_true = torch.zeros(1, self.sig_dim, device=self.device, dtype=torch.float64)
        
        empirical_sig = batched_calculate_signature_estimator(
            paths_batch, self.M, K, dummy_true, return_estimator=True
        ).squeeze(0)
        
        block_duration = T / K
        rescaling_time = 1.0 - block_duration
        
        # Initialize with smart initialization if not provided
        if init_params is None:
            theta_init, mu_init, sigma_init = self._compute_smart_initialization(path, K, T)
        else:
            theta_init, mu_init, sigma_init = init_params
        
        # For T=1, we focus on convergence over accuracy
        max_restarts = 1  # No restarts - always report convergence based on patience
        best_overall_loss = float('inf')
        best_overall_params = None
        overall_converged = False
        
        for restart in range(max_restarts):
            if restart > 0:
                # Add noise for restart
                noise_scale = 0.1 * (restart + 1)
                theta_init = theta_init + noise_scale * torch.randn_like(theta_init)
                mu_init = mu_init + noise_scale * torch.randn_like(mu_init)
                sigma_init = sigma_init + noise_scale * torch.randn_like(sigma_init)
                
                # Ensure positive definite after noise
                eigvals, eigvecs = torch.linalg.eigh(theta_init)
                eigvals = torch.clamp(eigvals, min=0.1)
                theta_init = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                
                eigvals, eigvecs = torch.linalg.eigh(sigma_init)
                eigvals = torch.clamp(eigvals, min=1e-3)
                sigma_init = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            
            theta = theta_init.clone().requires_grad_(True)
            mu = mu_init.clone().requires_grad_(True)
            
            STABILITY_EPS = 1e-6
            L_sigma = torch.linalg.cholesky(
                sigma_init + STABILITY_EPS * torch.eye(self.d, device=self.device, dtype=torch.float64)
            )
            L_sigma.requires_grad_(True)
            
            # Create unified optimizer (same for all K values for fairness)
            optimizer = ParameterInjector.create_unified_optimizer(
                [theta, mu, L_sigma], params, self.device
            )
            
            # Create unified scheduler (constant LR for fairness)
            scheduler = ParameterInjector.create_unified_scheduler(optimizer, params)
            
            # Convergence tracking
            best_loss = float('inf')
            best_params = None
            no_improve = 0
            converged = False
            
            for iteration in range(params.max_iterations):
                optimizer.zero_grad()
                
                sigma_reconstruct = L_sigma @ L_sigma.T
                
                # Expected signature and rescaling computations
                try:
                    expected_sig_block = compute_analytical_ou_expected_signature(
                        block_duration, self.d, self.M,
                        theta.unsqueeze(0),
                        mu.unsqueeze(0),
                        sigma_reconstruct.unsqueeze(0),
                        use_time_augmentation=False
                    ).squeeze(0)
                    
                    if rescaling_time > 1e-6:  # Only rescale if needed
                        L_matrix = construct_generator_matrix(
                            self.d, self.M,
                            theta.unsqueeze(0),
                            mu.unsqueeze(0),
                            sigma_reconstruct.unsqueeze(0),
                            use_time_augmentation=False
                        ).squeeze(0)
                        
                        rescaled_sig = taylor_exp_action(L_matrix, expected_sig_block, rescaling_time, order=20)
                    else:
                        rescaled_sig = expected_sig_block
                    
                except Exception as e:
                    # Stabilize parameters (ensure correct dtype)
                    theta_stable = theta + 0.1 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                    sigma_stable = sigma_reconstruct + 1e-3 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                    
                    expected_sig_block = compute_analytical_ou_expected_signature(
                        block_duration, self.d, self.M,
                        theta_stable.unsqueeze(0),
                        mu.unsqueeze(0),
                        sigma_stable.unsqueeze(0),
                        use_time_augmentation=False
                    ).squeeze(0)
                    
                    if rescaling_time > 1e-6:
                        L_matrix = construct_generator_matrix(
                            self.d, self.M,
                            theta_stable.unsqueeze(0),
                            mu.unsqueeze(0),
                            sigma_stable.unsqueeze(0),
                            use_time_augmentation=False
                        ).squeeze(0)
                        
                        rescaled_sig = taylor_exp_action(L_matrix, expected_sig_block, rescaling_time, order=20)
                    else:
                        rescaled_sig = expected_sig_block
                
                # Main loss with normalization
                diff = empirical_sig - rescaled_sig
                loss = torch.mean(diff ** 2)
                
                # Adaptive constraint penalties
                constraint_penalty = self._compute_constraint_penalty_jit(theta, sigma_reconstruct)
                
                # Scale constraint penalty based on T
                if T <= 1.0:
                    constraint_weight = 0.1
                else:
                    constraint_weight = 1.0
                
                total_loss = loss + constraint_weight * constraint_penalty
                
                total_loss.backward()
                
                # Track best parameters
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_params = (theta.detach().clone(), mu.detach().clone(), sigma_reconstruct.detach().clone())
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Check convergence with unified logic
                if ParameterInjector.validate_convergence_logic(
                    no_improve, params.patience, params
                ):
                    converged = True
                    break
                
                # Unified gradient clipping
                ParameterInjector.clip_gradients([theta, mu, L_sigma], params)
                
                optimizer.step()
                scheduler.step()
                
                # Smart periodic projection (like MLE) - only fix problematic parameters
                if iteration % 20 == 0:
                    with torch.no_grad():
                        # Quick check if projection is needed (fast diagonal check)
                        theta_diag = torch.diagonal(theta)
                        theta_needs_fix = torch.min(theta_diag) < 0.005
                        
                        if theta_needs_fix:
                            # Only project theta if actually problematic
                            try:
                                eigvals, eigvecs = torch.linalg.eigh(theta)
                                eigvals = torch.clamp(eigvals, min=0.05)
                                theta.data = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                            except:
                                theta.data = theta + 0.1 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                        
                        # Quick check for L_sigma (check diagonal of L matrix)
                        L_diag = torch.diagonal(L_sigma)
                        L_needs_fix = torch.min(L_diag) < 1e-5
                        
                        if L_needs_fix:
                            # Only project L_sigma if actually problematic
                            try:
                                sigma_temp = L_sigma @ L_sigma.T
                                eigvals_s, eigvecs_s = torch.linalg.eigh(sigma_temp)
                                eigvals_s = torch.clamp(eigvals_s, min=1e-3)
                                sigma_valid = eigvecs_s @ torch.diag(eigvals_s) @ eigvecs_s.T
                                L_sigma.data = torch.linalg.cholesky(
                                    sigma_valid + STABILITY_EPS * torch.eye(self.d, device=self.device)
                                )
                            except:
                                L_sigma.data = L_sigma + 1e-3 * torch.eye(self.d, device=self.device, dtype=torch.float64)
            
            # Check if this restart is better
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_params = best_params
                overall_converged = converged
                
                # If we converged with reasonable loss, stop restarts
                if converged and best_loss < 10.0:  # More realistic threshold
                    break
        
        # Final parameter handling
        if best_overall_params is None:
            sigma_final = L_sigma @ L_sigma.T
            best_overall_params = (theta.detach(), mu.detach(), sigma_final.detach())
        
        # Track optimization time
        elapsed = time.time() - start_time if verbose else 0.0
        if verbose:
            logger.info(f"Robust rescaled signature complete in {elapsed:.2f}s, converged={overall_converged}")
        
        # Return CalibrationResult with full tracking
        theta_final, mu_final, sigma_final = best_overall_params
        return CalibrationResult(
            theta=theta_final,
            mu=mu_final,
            sigma=sigma_final,
            converged=overall_converged,
            final_loss=best_overall_loss,
            iterations_used=iteration if 'iteration' in locals() else 0,
            parameters_used=params,
            method_name="Rescaled Signature",
            K_blocks=K,
            T_horizon=T,
            optimization_time=elapsed
        )


# =============================================================================
# OPTIMIZED ROBUST SIGNATURE CALIBRATOR (WITH CACHING)
# =============================================================================

# Try to import optimized functions, fall back gracefully if not available
try:
    from src.analytical.expected_signature_optimized import (
        compute_analytical_ou_expected_signature_cached,
        compute_batched_expected_signatures_optimized,
        clear_generator_cache,
        get_cache_stats
    )
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    logger.warning("Optimized signature functions not available, using standard version")


class OptimizedRobustSignatureCalibrator(RobustSignatureCalibrator):
    """
    Phase 5 optimized signature calibrator with batched K-value processing.
    
    Inherits from RobustSignatureCalibrator but overrides key methods for optimization.
    """
    
    def __init__(self, d, M, device, use_jit=False):
        super().__init__(d, M, device, use_jit=use_jit)
        self.use_caching = OPTIMIZATIONS_AVAILABLE
        logger.info(f"Optimized robust signature calibrator initialized (caching={'on' if self.use_caching else 'off'})") 
    
    def estimate(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Wrapper method for compatibility with existing experiment code.
        
        Returns 4-tuple (theta, mu, sigma, converged) as expected.
        """
        # Detach path to avoid backward graph issues
        path_detached = path.detach() if path.requires_grad else path
        
        # Use the optimized standard robust method with caching
        result = self.estimate_standard_robust(
            path=path_detached, K=K, T=T, 
            params=params, init_params=init_params, 
            verbose=verbose
        )
        
        # Return 4-tuple format expected by experiment
        # Detach tensors to avoid graph issues
        return result.theta.detach(), result.mu.detach(), result.sigma.detach(), result.converged
    
    def estimate_standard_robust(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Override to use cached expected signature computation if available.
        """
        if not OPTIMIZATIONS_AVAILABLE:
            # Fall back to parent implementation
            return super().estimate_standard_robust(path, K, T, params, init_params, verbose)
        
        # Get unified params
        params = self._get_unified_params(params)
        
        if verbose:
            logger.info(f"Optimized standard signature calibration for K={K}, T={T}...")
            start_time = time.time()
        
        # Compute empirical signature
        paths_batch = path.unsqueeze(0)
        dummy_true = torch.zeros(1, self.sig_dim, device=self.device, dtype=torch.float64)
        
        empirical_sig = batched_calculate_signature_estimator(
            paths_batch, self.M, K, dummy_true, return_estimator=True
        ).squeeze(0)
        
        block_duration = T / K
        
        # Initialize parameters
        if init_params is None:
            theta_init, mu_init, sigma_init = self._compute_smart_initialization(path, K, T)
        else:
            theta_init, mu_init, sigma_init = init_params
        
        # Optimization loop with cached generator
        theta = theta_init.clone().requires_grad_(True)
        mu = mu_init.clone().requires_grad_(True)
        
        STABILITY_EPS = 1e-6
        L_sigma = torch.linalg.cholesky(
            sigma_init + STABILITY_EPS * torch.eye(self.d, device=self.device, dtype=torch.float64)
        )
        L_sigma.requires_grad_(True)
        
        optimizer = ParameterInjector.create_unified_optimizer(
            [theta, mu, L_sigma], params, self.device
        )
        scheduler = ParameterInjector.create_unified_scheduler(optimizer, params)
        
        best_loss = float('inf')
        best_params = None
        no_improve = 0
        converged = False
        
        for iteration in range(params.max_iterations):
            optimizer.zero_grad()
            
            sigma_reconstruct = L_sigma @ L_sigma.T
            
            # USE CACHED COMPUTATION
            try:
                expected_sig = compute_analytical_ou_expected_signature_cached(
                    block_duration, self.d, self.M,
                    theta, mu, sigma_reconstruct,
                    use_time_augmentation=False,
                    use_cache=self.use_caching
                )
            except Exception as e:
                # Stabilize parameters
                theta_stable = theta + 0.1 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                sigma_stable = sigma_reconstruct + 1e-3 * torch.eye(self.d, device=self.device, dtype=torch.float64)
                expected_sig = compute_analytical_ou_expected_signature_cached(
                    block_duration, self.d, self.M,
                    theta_stable, mu, sigma_stable,
                    use_time_augmentation=False,
                    use_cache=self.use_caching
                )
            
            # Compute loss
            diff = empirical_sig - expected_sig
            loss = torch.mean(diff ** 2)
            
            # Add constraints
            constraint_penalty = self._compute_constraint_penalty_jit(theta, sigma_reconstruct)
            total_loss = loss + constraint_penalty
            
            total_loss.backward()
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = (theta.detach().clone(), mu.detach().clone(), sigma_reconstruct.detach().clone())
                no_improve = 0
            else:
                no_improve += 1
            
            # Check convergence
            if ParameterInjector.validate_convergence_logic(no_improve, params.patience, params):
                converged = True
                break
            
            # Gradient clipping
            ParameterInjector.clip_gradients([theta, mu, L_sigma], params)
            
            optimizer.step()
            scheduler.step()
        
        # Return best parameters
        if best_params is not None:
            theta_best, mu_best, sigma_best = best_params
        else:
            theta_best = theta.detach()
            mu_best = mu.detach()
            sigma_best = sigma_reconstruct.detach()
        
        if verbose:
            elapsed = time.time() - start_time
            logger.info(f"Optimization complete in {elapsed:.2f}s (converged: {converged})")
            
            if self.use_caching and OPTIMIZATIONS_AVAILABLE:
                stats = get_cache_stats()
                logger.info(f"Cache stats: Hit rate={stats['hit_rate']:.2%}")
        
        return CalibrationResult(
            theta=theta_best,
            mu=mu_best,
            sigma=sigma_best,
            converged=converged,
            final_loss=best_loss,
            iterations_used=iteration + 1,
            parameters_used=params,
            method_name='OptimizedRobustSignature',
            K_blocks=K,
            T_horizon=T,
            optimization_time=time.time() - start_time if verbose else 0.0
        )


# =============================================================================
# WRAPPER CALIBRATORS (FOR EXPERIMENT COMPATIBILITY)
# =============================================================================

class ExpectedSignatureCalibrator:
    """
    Standard expected signature calibrator for OU processes.
    
    This is a wrapper around RobustSignatureCalibrator that provides the
    expected signature method interface for the experiment.
    """
    
    def __init__(self, d, M, device):
        """
        Initialize the expected signature calibrator.
        
        Args:
            d: Dimension of the OU process
            M: Signature truncation level
            device: Torch device (cpu or cuda)
        """
        self.d = d
        self.M = M
        self.device = device
        self.robust_calibrator = RobustSignatureCalibrator(d, M, device, use_jit=False)
    
    def estimate(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Estimate OU parameters using expected signature matching.
        
        Args:
            path: Observed path
            K: Number of blocks
            T: Time horizon
            params: OptimizationParams (optional, uses defaults if None)
            init_params: Optional initial parameters (theta, mu, sigma) for warm-start
            verbose: Whether to print progress
            
        Returns:
            Tuple of (theta, mu, sigma, convergence_rate)
        """
        # Use robust calibrator's standard method with init_params support
        result = self.robust_calibrator.estimate_standard_robust(
            path, K, T, params=params, init_params=init_params, verbose=verbose
        )
        
        # Return in the format expected by the experiment
        return result.theta, result.mu, result.sigma, (1.0 if result.converged else 0.0)


class RescaledSignatureCalibrator:
    """
    Rescaled signature calibrator for OU processes.
    
    This is a wrapper around RobustSignatureCalibrator that provides the
    rescaled signature method interface for the experiment.
    """
    
    def __init__(self, d, M, device):
        """
        Initialize the rescaled signature calibrator.
        
        Args:
            d: Dimension of the OU process
            M: Signature truncation level
            device: Torch device (cpu or cuda)
        """
        self.d = d
        self.M = M
        self.device = device
        self.robust_calibrator = RobustSignatureCalibrator(d, M, device, use_jit=False)
    
    def estimate(self, path, K, T, params=None, init_params=None, verbose=False):
        """
        Estimate OU parameters using rescaled signature matching.
        
        Args:
            path: Observed path
            K: Number of blocks
            T: Time horizon
            params: OptimizationParams (optional, uses defaults if None)
            init_params: Optional initial parameters (theta, mu, sigma) for warm-start
            verbose: Whether to print progress
            
        Returns:
            Tuple of (theta, mu, sigma, convergence_rate)
        """
        # Use robust calibrator's rescaled method with init_params support
        result = self.robust_calibrator.estimate_rescaled_robust(
            path, K, T, params=params, init_params=init_params, verbose=verbose
        )
        
        # Return in the format expected by the experiment
        return result.theta, result.mu, result.sigma, (1.0 if result.converged else 0.0)


# Export all consolidated functionality
__all__ = [
    # Utility functions (from block_signature_calibration)
    'taylor_exp_action',
    'scaling_squaring_exp_action', 
    'compute_parameter_mse',
    
    # Calibrator classes
    'RobustSignatureCalibrator',
    'OptimizedRobustSignatureCalibrator',
    'ExpectedSignatureCalibrator',
    'RescaledSignatureCalibrator',
    'BatchedMLECalibrator'
]