"""
Method of Moments Calibrator for OU Processes

Implements parameter estimation using moment-matching approach, providing
a robust classical benchmark that doesn't require iterative optimization.

The method matches sample moments (mean, variance, autocorrelation) to their
theoretical counterparts for the OU process to derive parameter estimates.

Author: Bryson Schenck
Date: January 2025
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging
import time
from dataclasses import dataclass, field
from src.calibration_mega.parameter_management import (
    OptimizationParams, CalibrationResult, CalibratorInterface
)

logger = logging.getLogger(__name__)


class MethodOfMomentsCalibrator(CalibratorInterface):
    """
    Method of Moments calibrator for OU process parameters.
    
    This classical approach matches empirical moments to theoretical moments
    to estimate parameters without iterative optimization. It's particularly
    robust in high volatility regimes where likelihood-based methods may struggle.
    
    Key features:
    - Analytical solution via moment equations
    - K-value batching following Phase 1 pattern
    - GPU-accelerated tensor operations
    - Enhanced robustness through higher-order moments
    """
    
    def __init__(self, d: int, device: torch.device = None):
        """
        Initialize the Method of Moments calibrator.
        
        Args:
            d: Dimension of the OU process
            device: Torch device (cuda recommended for performance)
        """
        super().__init__(device)
        self.d = d
        
    def estimate(
        self,
        path: torch.Tensor,
        K: int,
        T: float,
        params: OptimizationParams = None,
        init_params: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        verbose: bool = False
    ) -> CalibrationResult:
        """
        Perform Method of Moments estimation with K-value batching.
        
        Args:
            path: Full path of the OU process [N x d]
            K: Number of blocks for batching
            T: Total time duration
            params: Optimization parameters (included for interface compliance)
            init_params: Not used (analytical method doesn't need initialization)
            verbose: Whether to print progress
            
        Returns:
            CalibrationResult with estimated parameters
        """
        start_time = time.time()
        
        # Get unified params if not provided (for interface compliance)
        if params is None:
            params = OptimizationParams()
        
        # Ensure path is on correct device
        if not isinstance(path, torch.Tensor):
            path = torch.tensor(path, device=self.device, dtype=torch.float32)
        else:
            path = path.to(self.device).float()
        
        N = len(path)
        dt = T / (N - 1)
        
        if verbose:
            logger.info(f"Method of Moments Calibration: K={K}, N={N}, dt={dt:.6f}")
        
        # Create K blocks following Phase 1 pattern
        steps_per_block = (N - 1) // K
        
        # Initialize batch storage
        theta_batch = torch.zeros(K, self.d, self.d, device=self.device)
        mu_batch = torch.zeros(K, self.d, device=self.device)
        sigma_batch = torch.zeros(K, self.d, self.d, device=self.device)
        converged_flags = torch.zeros(K, dtype=torch.bool, device=self.device)
        
        # Process each block
        for k in range(K):
            start_idx = k * steps_per_block
            end_idx = min(start_idx + steps_per_block + 1, N)
            block = path[start_idx:end_idx]
            
            # Compute moment-based estimates for this block
            theta_k, mu_k, sigma_k = self._compute_moments_params(block, dt)
            
            theta_batch[k] = theta_k
            mu_batch[k] = mu_k
            sigma_batch[k] = sigma_k
            
            # Check stability conditions
            converged_flags[k] = self._check_stability_conditions(
                theta_k, mu_k, sigma_k
            )
        
        # Average results across blocks (Phase 1 pattern)
        theta_avg = torch.mean(theta_batch, dim=0)
        mu_avg = torch.mean(mu_batch, dim=0)
        sigma_avg = torch.mean(sigma_batch, dim=0)
        
        # Compute convergence rate (percentage of stable blocks)
        convergence_rate = converged_flags.float().mean().item() * 100.0
        converged = convergence_rate >= 70.0  # Allow some failures in difficult regimes
        
        # Compute a pseudo "loss" for compatibility (moment matching error)
        final_loss = self._compute_moment_error(path, theta_avg, mu_avg, sigma_avg, dt).item()
        
        optimization_time = time.time() - start_time
        
        if verbose:
            logger.info(f"Method of Moments completed in {optimization_time:.2f}s")
            logger.info(f"Convergence rate: {convergence_rate:.1f}%")
            logger.info(f"Moment error: {final_loss:.6f}")
        
        return CalibrationResult(
            theta=theta_avg,
            mu=mu_avg,
            sigma=sigma_avg,
            converged=converged,
            final_loss=final_loss,
            iterations_used=1,  # Analytical method uses single computation
            parameters_used=params,
            method_name="Method_of_Moments",
            K_blocks=K,
            T_horizon=T,
            optimization_time=optimization_time,
            convergence_rate=convergence_rate  # Add convergence rate for tracking
        )
    
    def _compute_moments_params(
        self, 
        block: torch.Tensor, 
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Method of Moments parameter estimates for a single block.
        
        Matches sample moments to theoretical OU moments:
        - Mean: E[X_t] = μ
        - Variance: Var[X_t] = σ²/(2θ)
        - Autocorrelation: Corr[X_t, X_{t+Δt}] = exp(-θΔt)
        
        Args:
            block: Path segment [n_steps x d]
            dt: Time step
            
        Returns:
            Tuple of (theta, mu, sigma) estimates
        """
        n_steps = len(block) - 1
        
        if n_steps < 3:  # Need minimum observations
            # Return default stable parameters
            theta = torch.eye(self.d, device=self.device) * 0.5
            mu = torch.mean(block, dim=0)
            sigma = torch.eye(self.d, device=self.device) * 0.1
            return theta, mu, sigma
        
        # Compute sample moments
        mu_sample = torch.mean(block, dim=0)
        
        # Center the data
        block_centered = block - mu_sample
        
        # Sample variance-covariance matrix
        if self.d > 1:
            var_sample = torch.cov(block_centered.T)
        else:
            var_sample = torch.var(block_centered, unbiased=True).unsqueeze(0).unsqueeze(0)
        
        # Lag-1 autocorrelation (critical for theta estimation)
        X_lag0 = block_centered[:-1]
        X_lag1 = block_centered[1:]
        
        if self.d > 1:
            # Multivariate autocorrelation
            C_0 = torch.cov(X_lag0.T)
            C_1 = (X_lag0.T @ X_lag1) / (n_steps - 1)
            
            # Add regularization for numerical stability
            eps = 1e-6
            C_0_reg = C_0 + torch.eye(self.d, device=self.device) * eps
            
            try:
                # Autocorrelation matrix: R = C_0^{-1} @ C_1
                autocorr_matrix = torch.linalg.solve(C_0_reg, C_1)
                
                # Stabilize autocorrelation to valid range
                autocorr_matrix = self._stabilize_autocorrelation(autocorr_matrix)
                
                # Estimate theta from autocorrelation
                # R = exp(-theta * dt) => theta = -log(R) / dt
                # Note: torch doesn't have logm, use eigendecomposition
                eigenvals, eigenvecs = torch.linalg.eig(autocorr_matrix)
                log_eigenvals = torch.log(torch.clamp(torch.real(eigenvals), min=0.01))
                log_autocorr = eigenvecs @ torch.diag(log_eigenvals) @ torch.linalg.inv(eigenvecs)
                theta = -torch.real(log_autocorr) / dt
                
                # Ensure positive definite (mean-reverting)
                theta = self._ensure_positive_definite(theta, min_eig=0.001, max_eig=10.0)
                
            except Exception as e:
                # Fallback to diagonal estimation
                theta = torch.eye(self.d, device=self.device)
                for i in range(self.d):
                    x_i_lag0 = X_lag0[:, i]
                    x_i_lag1 = X_lag1[:, i]
                    if torch.std(x_i_lag0) > 1e-8 and torch.std(x_i_lag1) > 1e-8:
                        corr_i = torch.corrcoef(torch.stack([x_i_lag0, x_i_lag1]))[0, 1]
                        corr_i = torch.clamp(corr_i, min=0.01, max=0.99)
                        theta[i, i] = -torch.log(corr_i) / dt
                    else:
                        theta[i, i] = 0.5  # Default value
                theta = torch.clamp(theta, min=0.001, max=10.0)
                
        else:
            # Univariate case
            if torch.std(X_lag0) > 1e-8 and torch.std(X_lag1) > 1e-8:
                autocorr = torch.corrcoef(torch.stack([X_lag0.flatten(), X_lag1.flatten()]))[0, 1]
            else:
                autocorr = 0.5  # Default if no variation
            
            # Clamp to valid range
            autocorr = torch.clamp(autocorr, min=0.01, max=0.99)
            
            # Estimate theta
            theta = -torch.log(autocorr) / dt
            theta = torch.clamp(theta, min=0.001, max=10.0).unsqueeze(0).unsqueeze(0)
        
        # Estimate sigma from stationary variance relationship
        # Var_∞ = σσᵀ / (2θ) for stationary distribution
        # => σσᵀ = 2θ @ Var_sample
        try:
            sigma_squared = 2 * theta @ var_sample
            sigma = self._matrix_sqrt(sigma_squared)
        except:
            # Fallback to simple diagonal estimate
            sigma = torch.diag(torch.sqrt(torch.clamp(torch.diag(var_sample), min=1e-8)))
        
        # Use sample mean as mu estimate
        mu = mu_sample
        
        return theta, mu, sigma
    
    def _compute_moment_error(
        self,
        path: torch.Tensor,
        theta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Compute error between empirical and theoretical moments.
        
        This serves as a pseudo-loss for compatibility with optimization-based methods.
        
        Args:
            path: Full path [N x d]
            theta, mu, sigma: Parameter estimates
            dt: Time step
            
        Returns:
            Moment matching error
        """
        # Compute empirical moments
        emp_mean = torch.mean(path, dim=0)
        emp_var = torch.var(path, dim=0, unbiased=True)
        
        # Compute theoretical moments
        theo_mean = mu
        
        # Theoretical variance: σσᵀ / (2θ) for stationary distribution
        try:
            if self.d > 1:
                sigma_sigma_T = sigma @ sigma.T
                # For simplicity, use trace for scalar error
                theo_var = torch.diag(torch.linalg.solve(2 * theta, sigma_sigma_T))
            else:
                theo_var = (sigma ** 2) / (2 * theta).squeeze()
        except:
            theo_var = torch.ones_like(emp_var)
        
        # Compute relative errors
        mean_error = torch.norm(emp_mean - theo_mean) / (torch.norm(theo_mean) + 1e-8)
        var_error = torch.norm(emp_var - theo_var) / (torch.norm(theo_var) + 1e-8)
        
        # Compute autocorrelation error
        X_lag0 = path[:-1]
        X_lag1 = path[1:]
        
        if self.d == 1:
            emp_autocorr = torch.corrcoef(torch.stack([X_lag0.flatten(), X_lag1.flatten()]))[0, 1]
            theo_autocorr = torch.exp(-theta.squeeze() * dt)
            autocorr_error = torch.abs(emp_autocorr - theo_autocorr)
        else:
            # Use average diagonal autocorrelation for multivariate
            autocorr_errors = []
            for i in range(self.d):
                if torch.std(X_lag0[:, i]) > 1e-8 and torch.std(X_lag1[:, i]) > 1e-8:
                    emp_corr_i = torch.corrcoef(torch.stack([X_lag0[:, i], X_lag1[:, i]]))[0, 1]
                    theo_corr_i = torch.exp(-theta[i, i] * dt)
                    autocorr_errors.append(torch.abs(emp_corr_i - theo_corr_i))
            autocorr_error = torch.mean(torch.stack(autocorr_errors)) if autocorr_errors else torch.tensor(0.0)
        
        # Combined error (weighted sum)
        total_error = mean_error + var_error + autocorr_error
        
        return total_error
    
    def _check_stability_conditions(
        self,
        theta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        delta: float = 0.01,
        kappa_max: float = 100.0
    ) -> bool:
        """
        Check if parameters satisfy OU stability conditions.
        
        Args:
            theta: Drift matrix
            mu: Mean vector
            sigma: Volatility matrix
            delta: Minimum real part for eigenvalues
            kappa_max: Maximum condition number
            
        Returns:
            True if parameters are stable
        """
        try:
            # Check for NaN/Inf
            if (torch.isnan(theta).any() or torch.isinf(theta).any() or
                torch.isnan(mu).any() or torch.isinf(mu).any() or
                torch.isnan(sigma).any() or torch.isinf(sigma).any()):
                return False
            
            # 1. Stability: Re(eigenvalues(theta)) > delta
            eigenvals = torch.linalg.eigvals(theta)
            real_parts = torch.real(eigenvals)
            if (real_parts <= delta).any():
                return False
            
            # 2. Condition number: kappa(theta) < kappa_max
            real_parts_abs = torch.abs(real_parts)
            if real_parts_abs.min() > 0:
                condition_number = real_parts_abs.max() / real_parts_abs.min()
                if condition_number >= kappa_max:
                    return False
            else:
                return False
            
            # 3. Full rank volatility: rank(sigma) = d
            sigma_rank = torch.linalg.matrix_rank(sigma).item()
            if sigma_rank < self.d:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _stabilize_autocorrelation(
        self,
        autocorr: torch.Tensor,
        min_corr: float = 0.01,
        max_corr: float = 0.99
    ) -> torch.Tensor:
        """
        Stabilize autocorrelation matrix to valid range.
        
        Args:
            autocorr: Autocorrelation matrix
            min_corr: Minimum correlation value
            max_corr: Maximum correlation value
            
        Returns:
            Stabilized autocorrelation matrix
        """
        try:
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eig(autocorr)
            eigenvals = torch.real(eigenvals)
            
            # Clamp eigenvalues to valid correlation range
            eigenvals = torch.clamp(eigenvals, min=min_corr, max=max_corr)
            
            # Reconstruct
            autocorr_stable = eigenvecs @ torch.diag(eigenvals) @ torch.linalg.inv(eigenvecs)
            return torch.real(autocorr_stable)
        except:
            # Fallback to clamped diagonal
            return torch.diag(torch.clamp(torch.diag(autocorr), min=min_corr, max=max_corr))
    
    def _ensure_positive_definite(
        self,
        mat: torch.Tensor,
        min_eig: float = 0.001,
        max_eig: float = 10.0
    ) -> torch.Tensor:
        """
        Ensure matrix is positive definite with bounded eigenvalues.
        
        Args:
            mat: Input matrix
            min_eig: Minimum eigenvalue
            max_eig: Maximum eigenvalue
            
        Returns:
            Positive definite matrix
        """
        try:
            # Symmetrize
            mat_sym = (mat + mat.T) / 2
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(mat_sym)
            
            # Ensure positive eigenvalues
            eigenvals = torch.clamp(eigenvals, min=min_eig, max=max_eig)
            
            # Reconstruct
            mat_pd = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
            return mat_pd
        except:
            # Fallback to identity scaled by norm
            scale = torch.norm(mat).item() / self.d
            scale = max(min_eig, min(scale, max_eig))
            return torch.eye(self.d, device=self.device) * scale
    
    def _matrix_sqrt(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix square root for positive semi-definite matrix.
        
        Args:
            mat: Positive semi-definite matrix
            
        Returns:
            Square root matrix
        """
        try:
            # Ensure symmetric positive semi-definite
            mat_sym = (mat + mat.T) / 2
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(mat_sym)
            
            # Ensure non-negative eigenvalues
            eigenvals = torch.clamp(eigenvals, min=1e-8)
            
            # Compute square root
            sqrt_eigenvals = torch.sqrt(eigenvals)
            
            # Reconstruct
            mat_sqrt = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.T
            return mat_sqrt
        except:
            # Fallback to diagonal
            diag_elements = torch.clamp(torch.diag(mat), min=1e-8)
            return torch.diag(torch.sqrt(diag_elements))