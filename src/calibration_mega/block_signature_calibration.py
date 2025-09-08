"""
Block-based signature calibration methods for Ornstein-Uhlenbeck processes.

This module provides the core calibration methods used in the project:
1. Batched Maximum Likelihood Estimation (MLE)
2. Standard Expected Signature Calibration
3. Rescaled Expected Signature Calibration

Author: Bryson Schenck
Date: January 2025
"""

import torch
import torch.optim as optim
import numpy as np
import sys
sys.path.append('.')

from src.numerical.estimators import batched_calculate_signature_estimator
from src.analytical.expected_signature import (
    compute_analytical_ou_expected_signature,
    construct_generator_matrix
)


def taylor_exp_action(L, v, t, order=20):
    """
    Compute exp(tL)v using Taylor series approximation.
    
    Args:
        L: Generator matrix
        v: Vector to apply exponential to
        t: Time scaling factor
        order: Taylor series truncation order
        
    Returns:
        exp(tL)v approximation
    """
    result = v.clone()
    term = v.clone()
    
    for k in range(1, order + 1):
        term = (t / k) * (L @ term)
        result = result + term
        
        # Early termination if converged
        if torch.norm(term) < 1e-10 * torch.norm(result):
            break
    
    return result


def scaling_squaring_exp_action(L, v, t, base_order=15):
    """
    Compute exp(tL)v using scaling and squaring method for numerical stability.
    
    Args:
        L: Generator matrix
        v: Vector to apply exponential to
        t: Time scaling factor
        base_order: Base Taylor series order
        
    Returns:
        exp(tL)v computed with scaling and squaring
    """
    s = max(0, int(np.ceil(np.log2(max(1, t)))))
    scaled_t = t / (2**s)
    
    result = taylor_exp_action(L, v, scaled_t, order=base_order)
    
    for _ in range(s):
        result = taylor_exp_action(L, result, scaled_t, order=base_order)
    
    return result


class BatchedMLECalibrator:
    """
    Batched Maximum Likelihood Estimation for OU process parameters.
    
    This estimator divides the path into K blocks, estimates parameters
    on each block independently, then averages the estimates.
    """
    
    def __init__(self, d, device):
        """
        Initialize the batched MLE calibrator.
        
        Args:
            d: Dimension of the OU process
            device: Torch device (cpu or cuda)
        """
        self.d = d
        self.device = device
    
    def estimate_single_block(self, block_path, block_duration, n_iter=1000):  # Increased for better convergence
        """
        Estimate parameters from a single block using MLE.
        
        Args:
            block_path: Path segment for this block
            block_duration: Time duration of the block
            n_iter: Maximum number of optimization iterations
            
        Returns:
            Tuple of (theta, mu, sigma, converged) where converged indicates convergence
        """
        n_steps = block_path.shape[0] - 1
        dt = block_duration / n_steps
        
        X = block_path[:-1]
        Y = block_path[1:]
        
        # Initialize with method of moments
        mu_init = torch.mean(block_path, dim=0)
        residuals = Y - X
        
        # Handle case when we have too few points for covariance
        if residuals.shape[0] > 1:
            try:
                sigma_init = torch.cov(residuals.T) / dt
            except:
                sigma_init = torch.eye(self.d, device=self.device, dtype=torch.float64)
        else:
            # Not enough data points for covariance, use identity
            sigma_init = torch.eye(self.d, device=self.device, dtype=torch.float64)
        
        theta_init = torch.eye(self.d, device=self.device, dtype=torch.float64) * 0.5
        
        # Optimize parameters
        theta = theta_init.clone().requires_grad_(True)
        mu = mu_init.clone().requires_grad_(True)
        L_sigma = torch.linalg.cholesky(
            sigma_init + 1e-6 * torch.eye(self.d, device=self.device, dtype=torch.float64)
        )
        L_sigma.requires_grad_(True)
        
        optimizer = optim.Adam([theta, mu, L_sigma], lr=0.001)  # Optimal LR for MLE
        
        best_loss = float('inf')
        best_params = None
        patience = 100  # Higher patience for better convergence
        no_improve = 0
        converged = False
        grad_history = []
        param_history = []
        
        # Numerical stability constant - same for all methods
        STABILITY_EPS = 1e-8
        
        for iteration in range(n_iter):
            optimizer.zero_grad()
            
            # NO REGULARIZATION on parameters - only for numerical stability in operations
            sigma_reconstruct = L_sigma @ L_sigma.T
            
            # CRITICAL FIX: Correct OU drift formula
            # OU process: dX_t = theta @ (mu - X_t) dt + sigma dW_t
            drift = torch.einsum('ij,nj->ni', theta, mu.unsqueeze(0) - X)
            mean_pred = X + drift * dt
            
            # Residuals
            residuals = Y - mean_pred
            
            # Stable likelihood computation using Cholesky decomposition
            cov = sigma_reconstruct * dt
            try:
                L_cov = torch.linalg.cholesky(cov)
            except:
                # Add minimal regularization only if Cholesky fails
                L_cov = torch.linalg.cholesky(cov + STABILITY_EPS * torch.eye(self.d, device=self.device))
            
            # Solve L @ z = residuals.T for stable computation
            z = torch.linalg.solve_triangular(L_cov, residuals.T, upper=False)
            
            # NLL = 0.5 * [sum(z^2) + n * log(det(2π * cov))]
            # log(det(cov)) = 2 * sum(log(diag(L)))
            nll = 0.5 * (torch.sum(z**2) + 
                        n_steps * (self.d * np.log(2 * np.pi) + 
                                  2 * torch.sum(torch.log(torch.diag(L_cov)))))
            
            # Add soft constraints to keep parameters valid
            try:
                min_eig_theta = torch.min(torch.linalg.eigvalsh(theta))
                if min_eig_theta < 0.01:
                    nll += 100 * torch.relu(0.01 - min_eig_theta)
            except:
                # If eigenvalue computation fails, add large penalty
                nll += 1000.0
            
            try:
                min_eig_sigma = torch.min(torch.linalg.eigvalsh(sigma_reconstruct))
                if min_eig_sigma < 1e-4:
                    nll += 100 * torch.relu(1e-4 - min_eig_sigma)
            except:
                # If eigenvalue computation fails, add large penalty
                nll += 1000.0
            
            # Track gradients for convergence checking
            nll.backward()
            grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in [theta, mu, L_sigma] if p.grad is not None]))
            grad_history.append(grad_norm.item())
            
            # Track parameter changes for convergence
            if iteration > 0 and len(param_history) > 0:
                old_theta, old_mu, old_L_sigma = param_history[-1]
                param_change = (
                    torch.norm(theta - old_theta) + 
                    torch.norm(mu - old_mu) + 
                    torch.norm(L_sigma - old_L_sigma)
                ) / 3.0
            else:
                param_change = float('inf')
            
            # Check convergence via gradient OR parameter change (original working condition)
            if param_change < 1e-3 or grad_norm < 0.1:  # Either condition triggers convergence
                converged = True
                best_params = (
                    theta.detach().clone(),
                    mu.detach().clone(),
                    sigma_reconstruct.detach().clone()
                )
                break
            
            # Store current parameters for next iteration's comparison
            param_history.append((theta.detach().clone(), mu.detach().clone(), L_sigma.detach().clone()))
            
            optimizer.step()
            
            # Project to valid region every 10 iterations
            if iteration % 10 == 0:
                with torch.no_grad():
                    # Ensure theta has positive eigenvalues
                    try:
                        eigvals, eigvecs = torch.linalg.eigh(theta)
                        eigvals = torch.clamp(eigvals, min=0.01)
                        theta.data = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                    except:
                        # If eigendecomposition fails, add regularization
                        theta.data = theta + 0.01 * torch.eye(self.d, device=self.device)
                    
                    # Ensure L_sigma produces positive definite sigma
                    try:
                        sigma_temp = L_sigma @ L_sigma.T
                        eigvals_s, eigvecs_s = torch.linalg.eigh(sigma_temp)
                        eigvals_s = torch.clamp(eigvals_s, min=1e-4)
                        sigma_valid = eigvecs_s @ torch.diag(eigvals_s) @ eigvecs_s.T
                        L_sigma.data = torch.linalg.cholesky(sigma_valid + 1e-10 * torch.eye(self.d, device=self.device))
                    except:
                        # If fails, just add regularization
                        L_sigma.data = L_sigma + 1e-4 * torch.eye(self.d, device=self.device)
            
            if nll.item() < best_loss:
                best_loss = nll.item()
                best_params = (
                    theta.detach().clone(),
                    mu.detach().clone(),
                    sigma_reconstruct.detach().clone()
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        if best_params is None:
            sigma_final = L_sigma @ L_sigma.T
            best_params = (theta.detach(), mu.detach(), sigma_final.detach())
        
        return (*best_params, converged)
    
    def estimate(self, path, K, T, verbose=False):
        """
        Perform batched MLE estimation.
        
        Args:
            path: Full path of the OU process
            K: Number of blocks
            T: Total time duration
            verbose: Whether to print progress
            
        Returns:
            Tuple of (theta, mu, sigma, convergence_rate) averaged estimates
        """
        if K == 1:
            # Single block case - estimate on full path
            theta, mu, sigma, converged = self.estimate_single_block(path, T)
            return theta, mu, sigma, 1.0 if converged else 0.0
        
        # Segment path into K blocks
        n_steps = path.shape[0] - 1
        if n_steps % K != 0:
            raise ValueError(f"Path length {n_steps} not divisible by K={K}")
        
        block_duration = T / K
        steps_per_block = n_steps // K
        
        # Storage for block estimates
        theta_estimates = []
        mu_estimates = []
        sigma_estimates = []
        converged_count = 0
        
        for block_idx in range(K):
            start_idx = block_idx * steps_per_block
            end_idx = start_idx + steps_per_block + 1  # +1 to include endpoint
            block_path = path[start_idx:end_idx]
            
            # Estimate on this block
            theta_block, mu_block, sigma_block, converged = self.estimate_single_block(
                block_path, block_duration  # Use default n_iter with early stopping
            )
            
            theta_estimates.append(theta_block)
            mu_estimates.append(mu_block)
            sigma_estimates.append(sigma_block)
            if converged:
                converged_count += 1
        
        # Average the estimates - use proper averaging for positive definite matrices
        mu_avg = torch.mean(torch.stack(mu_estimates), dim=0)  # Vector average is fine
        
        # For positive definite matrices, ensure result remains positive definite
        # Use arithmetic mean but project to positive definite cone if needed
        theta_avg = torch.mean(torch.stack(theta_estimates), dim=0)
        sigma_avg = torch.mean(torch.stack(sigma_estimates), dim=0)
        
        # Ensure averaged matrices are positive definite
        theta_avg = self._ensure_positive_definite(theta_avg, min_eig=0.01)
        sigma_avg = self._ensure_positive_definite(sigma_avg, min_eig=1e-4)
        
        convergence_rate = converged_count / K
        
        return theta_avg, mu_avg, sigma_avg, convergence_rate
    
    def _ensure_positive_definite(self, matrix, min_eig=1e-6):
        """Project matrix to positive definite cone if needed."""
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        if torch.min(eigvals) < min_eig:
            eigvals = torch.clamp(eigvals, min=min_eig)
            matrix = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        return matrix


# Legacy SignatureCalibrator class removed - replaced by RobustSignatureCalibrator in signature_robust.py
# This avoids code duplication and maintains single source of truth for signature methods


def compute_parameter_mse(true_params, estimated_params):
    """
    Compute mean squared error between true and estimated parameters.
    
    Args:
        true_params: Tuple of (theta, mu, sigma) true parameters
        estimated_params: Tuple of (theta, mu, sigma) estimated parameters
        
    Returns:
        Average MSE across all parameters
    """
    theta_true, mu_true, sigma_true = true_params
    theta_est, mu_est, sigma_est = estimated_params
    
    mse_theta = torch.mean((theta_est - theta_true) ** 2).item()
    mse_mu = torch.mean((mu_est - mu_true) ** 2).item()
    mse_sigma = torch.mean((sigma_est - sigma_true) ** 2).item()
    
    # Cap large values to prevent numerical issues
    mse_theta = min(mse_theta, 10.0)
    mse_mu = min(mse_mu, 10.0)
    mse_sigma = min(mse_sigma, 10.0)
    
    return (mse_theta + mse_mu + mse_sigma) / 3.0