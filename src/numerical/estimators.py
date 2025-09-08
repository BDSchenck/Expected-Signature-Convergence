import torch
import signatory
import numpy as np

def batched_segment_path(paths, K_N, n_steps_per_block_override=None):
    """
    Segments a batch of long paths into non-overlapping blocks.
    
    Args:
        paths (torch.Tensor): A batch of paths of shape (n_sims, total_steps, d_path).
        K_N (int): The number of blocks to segment each path into.

    Returns:
        torch.Tensor: A tensor of shape (n_sims * K_N, steps_per_block + 1, d_path).
    """
    n_sims, total_steps, d_path = paths.shape
    if n_steps_per_block_override is not None:
        steps_per_block = n_steps_per_block_override
    else:
        steps_per_block = (total_steps - 1) // K_N

    if (total_steps - 1) % K_N != 0:
        raise ValueError(
            f"Total steps in the path ({(total_steps - 1)}) cannot be evenly divided by K_N ({K_N})."
        )

    # Create a view of the paths tensor that we can index into
    # Shape: (n_sims, K_N, steps_per_block, d_path)
    strided_paths = paths.as_strided(
        size=(n_sims, K_N, steps_per_block + 1, d_path),
        stride=(total_steps * d_path, steps_per_block * d_path, d_path, 1),
        storage_offset=0
    )
    
    # Reshape to the final desired output
    return strided_paths.reshape(n_sims * K_N, steps_per_block + 1, d_path)

def batched_calculate_signature_estimator(paths, M, K_N, true_expected_sig, n_steps_per_block_override=None, return_estimator=False):
    """
    Calculates the block-based signature estimator for a batch of long paths.
    If return_estimator is False, it returns the mean squared error for the batch.
    If return_estimator is True, it returns the estimator tensor itself.
    """
    n_sims = paths.shape[0]
    
    # 1. Segment all paths into blocks in one go
    if n_steps_per_block_override:
        steps_per_block = n_steps_per_block_override
    else:
        steps_per_block = (paths.shape[1] - 1) // K_N
    
    batched_blocks = batched_segment_path(paths, K_N, steps_per_block)

    # 2. Manually normalize the blocks to start at the origin (vectorized)
    start_points = batched_blocks[:, 0, :].unsqueeze(1)
    normalized_blocks = batched_blocks - start_points

    # 3. Compute all signatures in one go
    block_sigs_no_0 = signatory.signature(normalized_blocks, M)
    block_sigs = torch.cat([torch.ones(block_sigs_no_0.shape[0], 1, device=block_sigs_no_0.device), block_sigs_no_0], dim=1)

    # 4. Reshape to separate simulations from blocks and compute the mean estimator
    sig_dim = block_sigs.shape[-1]
    block_sigs_reshaped = block_sigs.view(n_sims, K_N, sig_dim)
    estimators = torch.mean(block_sigs_reshaped, dim=1)

    if return_estimator:
        return estimators

    # 5. Calculate the Squared Error for each simulation in the batch
    squared_errors = torch.sum((estimators - true_expected_sig)**2, dim=1)

    # 6. Return the mean of the squared errors across the batch
    return torch.mean(squared_errors)