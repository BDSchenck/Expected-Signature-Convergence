import torch
import numpy as np

def generate_valid_ou_parameters(d, device):
    """Generates a single set of valid, random OU parameters."""
    while True:
        theta = (torch.rand(d, d, device=device, dtype=torch.float64) * 2 - 1)
        eigenvalues = torch.linalg.eigvals(theta)
        if torch.all(eigenvalues.real > 0):
            break

    mu = (torch.rand(d, device=device, dtype=torch.float64) * 2 - 1)
    A = (torch.rand(d, d, device=device, dtype=torch.float64) * 2 - 1)
    sigma = (A @ A.T + torch.eye(d, device=device, dtype=torch.float64) * 1e-2) * 3

    # print("Generated Random Valid OU Parameters:")
    # print(f"  theta:\n{theta}\n")
    # print(f"  mu:\n{mu}\n")
    # print(f"  sigma:\n{sigma}\n")")

    return theta, mu, sigma

def generate_ou_parameters_for_regime(d, device, regime_config, num_batches=1):
    """
    Generates a batch of OU parameters based on a specified regime configuration.

    Args:
        d (int): Dimension of the process.
        device (torch.device): Device to place tensors on.
        regime_config (dict): A dictionary specifying the parameter regime.
        num_batches (int): Number of parameter sets to generate.

    Returns:
        tuple: (thetas, mus, sigmas) as batched tensors of shape (num_batches, ...) 
               or (1, ...) if num_batches is 1.
    """
    # Generate theta matrix from specified eigenvalues
    eigenvalue_min, eigenvalue_max = regime_config['theta_eigenvalue_range']

    # Generate num_batches of random symmetric matrices
    A = torch.randn(num_batches, d, d, device=device, dtype=torch.float64)
    # Make them symmetric positive definite
    theta_base = A @ A.transpose(-1, -2)

    # Compute eigenvalues and eigenvectors for each matrix in the batch
    eigvals_base, eigvecs_base = torch.linalg.eigh(theta_base)

    # Scale eigenvalues to the desired range
    min_eig_base = eigvals_base.min(dim=-1, keepdim=True).values
    max_eig_base = eigvals_base.max(dim=-1, keepdim=True).values
    
    range_eig_base = max_eig_base - min_eig_base
    # Handle cases where range_eig_base is zero (all eigenvalues are the same)
    normalized_eigvals = torch.where(range_eig_base == 0, torch.zeros_like(eigvals_base), (eigvals_base - min_eig_base) / range_eig_base)

    eigenvalue_min_tensor = torch.tensor(eigenvalue_min, device=device, dtype=torch.float64)
    eigenvalue_max_tensor = torch.tensor(eigenvalue_max, device=device, dtype=torch.float64)
    scaled_eigenvalues = normalized_eigvals * (eigenvalue_max_tensor - eigenvalue_min_tensor) + eigenvalue_min_tensor

    D_scaled = torch.diag_embed(scaled_eigenvalues)
    theta_batch = eigvecs_base @ D_scaled @ eigvecs_base.transpose(-1, -2)

    # Generate mu vector
    mu_min, mu_max = regime_config['mu_range']
    mu_batch = torch.rand(num_batches, d, device=device, dtype=torch.float64) * (mu_max - mu_min) + mu_min

    # Generate sigma matrix
    diag_min, diag_max = regime_config['sigma_diag_range']
    corr_level_config = regime_config['sigma_corr_level']
    
    # Handle both single value and range for correlation level
    if isinstance(corr_level_config, (list, tuple)) and len(corr_level_config) == 2:
        # Random correlation level for each batch element
        corr_min, corr_max = corr_level_config
        corr_level = torch.rand(num_batches, 1, 1, device=device, dtype=torch.float64) * (corr_max - corr_min) + corr_min
    else:
        # Single correlation level for all batch elements
        corr_level = torch.tensor(corr_level_config, device=device, dtype=torch.float64).view(1, 1, 1).expand(num_batches, 1, 1)
    
    A_sigma = torch.randn(num_batches, d, d, device=device, dtype=torch.float64)
    S_base_sigma = A_sigma @ A_sigma.transpose(-1, -2)
    
    diag_elements_sigma = torch.sqrt(torch.rand(num_batches, d, device=device, dtype=torch.float64) * (diag_max - diag_min) + diag_min)
    diag_S_base_sigma = torch.diagonal(S_base_sigma, dim1=-2, dim2=-1)
    
    # Handle potential division by zero if diag_S_base_sigma has zeros
    diag_scaler_elements_sigma = torch.where(diag_S_base_sigma == 0, torch.zeros_like(diag_elements_sigma), diag_elements_sigma / torch.sqrt(diag_S_base_sigma))
    
    diag_scaler_matrix_sigma = torch.diag_embed(diag_scaler_elements_sigma)

    S_scaled_sigma = diag_scaler_matrix_sigma @ S_base_sigma @ diag_scaler_matrix_sigma

    identity_batch = torch.eye(d, device=device, dtype=torch.float64).unsqueeze(0).repeat(num_batches, 1, 1)
    
    diag_S_scaled_sigma = torch.diagonal(S_scaled_sigma, dim1=-2, dim2=-1)
    diag_S_scaled_matrix_sigma = torch.diag_embed(diag_S_scaled_sigma)

    sigma_batch = (1 - corr_level) * diag_S_scaled_matrix_sigma + corr_level * S_scaled_sigma
    sigma_batch += identity_batch * 1e-4

    if num_batches == 1:
        # print(f"Generated OU Parameters for Regime: {regime_config}")
        # print(f"  theta (eigenvalues: {torch.linalg.eigvalsh(theta_batch.squeeze(0))[0]}):\n{theta_batch.squeeze(0)}\n")
        # print(f"  mu:\n{mu_batch.squeeze(0)}\n")
        # print(f"  sigma:\n{sigma_batch.squeeze(0)}\n")
        pass

    return theta_batch, mu_batch, sigma_batch