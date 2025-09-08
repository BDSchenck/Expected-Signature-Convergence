"""
Phase 5 Optimized Expected Signature Computation
================================================

Advanced GPU optimization with generator matrix caching and batched operations.

Key Optimizations:
1. Generator matrix caching - reuse structure across K values
2. Batched matrix exponential computation
3. Memory-efficient tensor operations
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from src.analytical.expected_signature import (
    get_tensor_basis,
    construct_generator_matrix as construct_generator_original
)


class GeneratorMatrixCache:
    """
    Cache for generator matrices to avoid redundant construction.
    
    The key insight: Generator structure is parameter-dependent but not K-dependent.
    We can cache the generator and only scale it by different time values.
    """
    
    def __init__(self, device='cuda'):
        self.cache = {}
        self.device = device
        self.hit_count = 0
        self.miss_count = 0
        
    def _compute_cache_key(self, d: int, M: int, theta: torch.Tensor, 
                           mu: torch.Tensor, sigma: torch.Tensor) -> str:
        """Create a cache key from parameters."""
        # Use parameter shapes and values for cache key
        # Round to 3 decimal places for optimal balance of accuracy and cache efficiency
        # Parameters that differ by <0.001 are considered the same for caching
        theta_rounded = torch.round(theta * 1000) / 1000  # Round to 3 decimal places
        mu_rounded = torch.round(mu * 1000) / 1000
        sigma_rounded = torch.round(sigma * 1000) / 1000
        
        key = f"{d}_{M}_{theta_rounded.flatten().tolist()}_{mu_rounded.flatten().tolist()}_{sigma_rounded.flatten().tolist()}"
        return key
    
    def get_generator(self, d: int, M: int, theta: torch.Tensor, 
                     mu: torch.Tensor, sigma: torch.Tensor,
                     use_time_augmentation: bool = False) -> torch.Tensor:
        """Get generator matrix from cache or compute if not present."""
        key = self._compute_cache_key(d, M, theta, mu, sigma)
        
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            # Compute generator matrix
            generator = construct_generator_original(
                d, M, 
                theta.unsqueeze(0) if theta.dim() == 2 else theta,
                mu.unsqueeze(0) if mu.dim() == 1 else mu,
                sigma.unsqueeze(0) if sigma.dim() == 2 else sigma,
                use_time_augmentation
            )
            
            # Cache it - DETACH to avoid storing computation graph
            self.cache[key] = generator.detach()
            return generator
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


# Global cache instance
_GENERATOR_CACHE = GeneratorMatrixCache()


def compute_analytical_ou_expected_signature_cached(
    t: float, d: int, M: int, 
    theta: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    use_time_augmentation: bool = False,
    use_cache: bool = True
) -> torch.Tensor:
    """
    Optimized expected signature computation with generator caching.
    
    Args:
        t: Time duration
        d: Dimension of the process
        M: Signature truncation level
        theta: Mean-reversion matrix
        mu: Long-term mean vector
        sigma: Volatility matrix
        use_time_augmentation: Whether to use time augmentation
        use_cache: Whether to use generator caching
        
    Returns:
        Expected signature tensor
    """
    device = theta.device
    
    # Get generator matrix (from cache if available)
    if use_cache:
        L = _GENERATOR_CACHE.get_generator(d, M, theta, mu, sigma, use_time_augmentation)
    else:
        L = construct_generator_original(
            d, M,
            theta.unsqueeze(0) if theta.dim() == 2 else theta,
            mu.unsqueeze(0) if mu.dim() == 1 else mu,
            sigma.unsqueeze(0) if sigma.dim() == 2 else sigma,
            use_time_augmentation
        )
    
    # Handle batched or single input
    if L.dim() == 3:  # Batched
        batch_size = L.shape[0]
        L = L.squeeze(0) if batch_size == 1 else L
    
    # Scale by time and compute matrix exponential
    if L.dim() == 2:  # Single matrix
        exp_tL = torch.linalg.matrix_exp(t * L)
    else:  # Batched matrices
        exp_tL = torch.linalg.matrix_exp(t * L)
    
    # Extract expected signature (first column)
    dim_path = d + 1 if use_time_augmentation else d
    sig_dim = sum(dim_path**i for i in range(M+1))
    
    if exp_tL.dim() == 2:
        expected_sig = exp_tL[:, 0]
    else:
        expected_sig = exp_tL[:, :, 0]
    
    return expected_sig


def compute_batched_expected_signatures_optimized(
    K_values: List[int], T: float, d: int, M: int,
    theta: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    use_time_augmentation: bool = False
) -> Dict[int, torch.Tensor]:
    """
    Compute expected signatures for multiple K values in a batched manner.
    
    This is the key optimization: instead of sequential processing,
    we batch K values with similar computational requirements.
    
    Args:
        K_values: List of K values to compute
        T: Total time horizon
        d: Dimension
        M: Truncation level
        theta, mu, sigma: OU parameters
        use_time_augmentation: Time augmentation flag
        
    Returns:
        Dictionary mapping K to expected signature
    """
    device = theta.device
    results = {}
    
    # Get single generator matrix (cached)
    L = _GENERATOR_CACHE.get_generator(d, M, theta, mu, sigma, use_time_augmentation)
    if L.dim() == 3:
        L = L.squeeze(0)  # Remove batch dimension if present
    
    # Group K values by computational complexity
    small_k = [k for k in K_values if k <= 4096]
    medium_k = [k for k in K_values if 4096 < k <= 65536]
    large_k = [k for k in K_values if k > 65536]
    
    # Process each group
    for k_group in [small_k, medium_k, large_k]:
        if not k_group:
            continue
            
        # Prepare time values for this group
        block_durations = torch.tensor([T / k for k in k_group], 
                                      dtype=torch.float64, device=device)
        
        # Batch compute matrix exponentials
        # Stack scaled matrices
        scaled_matrices = torch.stack([t * L for t in block_durations])
        
        # Compute all exponentials at once
        exp_matrices = torch.linalg.matrix_exp(scaled_matrices)
        
        # Extract expected signatures
        for i, k in enumerate(k_group):
            results[k] = exp_matrices[i, :, 0]
    
    return results


def clear_generator_cache():
    """Clear the global generator cache."""
    global _GENERATOR_CACHE
    _GENERATOR_CACHE.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get generator cache statistics."""
    global _GENERATOR_CACHE
    return _GENERATOR_CACHE.get_stats()