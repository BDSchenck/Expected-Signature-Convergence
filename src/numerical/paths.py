"""
This module contains functions for generating stochastic paths.
"""
import torch

def generate_ou_process(num_paths, n_steps, T, d, theta, mu, sigma, device, use_time_augmentation=False, H=0.5):
    """
    Generate a batch of d-dimensional Ornstein-Uhlenbeck paths.

    Args:
        num_paths (int): Number of paths to generate (batch size).
        n_steps (int): Number of time steps for each path.
        T (float): Total time duration for each path.
        d (int): Dimension of the OU process.
        theta (torch.Tensor): Mean-reversion matrix (num_paths, d, d).
        mu (torch.Tensor): Long-term mean vector (num_paths, d).
        sigma (torch.Tensor): Volatility matrix (num_paths, d, d).
        device (torch.device): Device to place tensors on.
        use_time_augmentation (bool): If True, prepends a time channel to the path.
        H (float): Hurst parameter for the driving fractional Brownian motion.
    """
    dt = T / n_steps
    paths = torch.zeros(num_paths, n_steps + 1, d, device=device, dtype=torch.float64)
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device, dtype=torch.float64))

    for i in range(n_steps):
        dW = torch.randn(num_paths, d, device=device, dtype=torch.float64) * sqrt_dt
        
        drift = torch.bmm(theta, (mu.expand(num_paths, d) - paths[:, i, :]).unsqueeze(-1)).squeeze(-1)
        diffusion = torch.bmm(sigma, dW.unsqueeze(-1)).squeeze(-1)
        
        paths[:, i + 1, :] = paths[:, i, :] + drift * dt + diffusion

    if use_time_augmentation:
        time_vec = torch.linspace(0, T, n_steps + 1, device=device, dtype=torch.float64).unsqueeze(0).unsqueeze(2)
        time_vec = time_vec.repeat(num_paths, 1, 1)
        paths = torch.cat([time_vec, paths], dim=2)

    return paths