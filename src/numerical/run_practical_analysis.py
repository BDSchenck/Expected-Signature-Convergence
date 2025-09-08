# ==============================================================================
# == Experiment 2: Practical Data Analysis ("Fixed-Length Path")
# ==============================================================================
#
# PURPOSE:
# This script implements the "Fixed-Length Path" experiment. This is the most
# practical setup for real-world data analysis and is the correct design to
# investigate the bias-variance trade-off (the "U-shaped curve").
#
# METHODOLOGY:
#   - A Monte Carlo simulation is run where a new path is generated for each run.
#   - The independent variable `K` is the number of blocks the path is divided into.
#   - The total path length and total number of discrete steps are FIXED.
#   - The estimator for a block is compared to the analytical truth for that
#     same block duration (fast, unscaled comparison).
#
# ==============================================================================

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from .paths import generate_ou_process
from .estimators import batched_calculate_signature_estimator
from ..analytical.expected_signature import compute_analytical_ou_expected_signature
from .parameters import generate_ou_parameters_for_regime
from .statistical_analysis import compute_slope_with_confidence_interval, format_slope_result

def run_practical_analysis_experiment(device):
    print("\n\n===== Starting Experiment 2: Practical Analysis ('Fixed-Length Path') ======")
    start_time = time.time()

    # --- Fixed Parameters for the Entire Experiment ---
    d = 2
    M = 4
    total_duration = 100.0  # Fixed dataset length of T=100
    total_steps = 8192  # Reduced for computational feasibility
    n_monte_carlo_sims = 100  # Fixed Monte Carlo simulations
    batch_size = 100 # Process in batches of 100
    
    # K values as specified in the tex file
    K_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    regime_configs = {
        "Slow Reversion, Low Volatility": {
            'theta_eigenvalue_range': [0.05, 0.2], 'mu_range': [-1, 1],
            'sigma_diag_range': [0.1, 0.3], 'sigma_corr_level': [0.0, 1.0]  # Randomized
        },
        "Fast Reversion, Low Volatility": {
            'theta_eigenvalue_range': [1.0, 3.0], 'mu_range': [-1, 1],
            'sigma_diag_range': [0.1, 0.3], 'sigma_corr_level': [0.0, 1.0]  # Randomized
        },
        "Slow Reversion, High Volatility": {
            'theta_eigenvalue_range': [0.05, 0.2], 'mu_range': [-1, 1],
            'sigma_diag_range': [1.0, 2.0], 'sigma_corr_level': [0.0, 1.0]  # Randomized
        },
        "Fast Reversion, High Volatility": {
            'theta_eigenvalue_range': [1.0, 3.0], 'mu_range': [-1, 1],
            'sigma_diag_range': [1.0, 2.0], 'sigma_corr_level': [0.0, 1.0]  # Randomized
        }
    }

    all_results = []
    summary_results = []
    fig, ax = plt.subplots(figsize=(12, 7))
    highest_initial_error = 0

    for regime_name, config in regime_configs.items():
        print(f"\n--- Running Regime: {regime_name} ---")
        
        mse_vs_K = []
        valid_K_values = [k for k in K_values if total_steps % k == 0]
        
        for idx, K_num_blocks in enumerate(valid_K_values):
            print(f"  K (num_blocks)={K_num_blocks}, Monte Carlo sims={n_monte_carlo_sims}...")
            block_duration = total_duration / K_num_blocks
            n_steps_per_block = total_steps // K_num_blocks
            
            # Process in batches for memory efficiency
            mse_accumulator = 0.0
            n_batches = (n_monte_carlo_sims + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_monte_carlo_sims)
                batch_sims = batch_end - batch_start
                
                # Generate parameters and paths for this batch
                thetas, mus, sigmas = generate_ou_parameters_for_regime(d, device, config, batch_sims)
                paths = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, False)
                
                # Compute MSE for this batch
                true_sigs = compute_analytical_ou_expected_signature(block_duration, d, M, thetas, mus, sigmas, False)
                mse_batch = batched_calculate_signature_estimator(paths, M, K_num_blocks, true_sigs, n_steps_per_block).item()
                mse_accumulator += mse_batch * batch_sims
                
                # Clean up GPU memory
                del paths, thetas, mus, sigmas, true_sigs
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # Average MSE across all simulations
            mse = mse_accumulator / n_monte_carlo_sims
            mse_vs_K.append(mse)

        if mse_vs_K[0] > highest_initial_error:
            highest_initial_error = mse_vs_K[0]

        # Compute slope with confidence intervals and statistical tests
        stats_result = compute_slope_with_confidence_interval(np.array(valid_K_values), np.array(mse_vs_K))
        
        print(f"    {format_slope_result(stats_result)}")
        
        summary_results.append({
            'regime': regime_name,
            'empirical_slope': stats_result['slope'],
            'slope_ci_lower': stats_result['slope_ci_lower'],
            'slope_ci_upper': stats_result['slope_ci_upper'],
            'slope_stderr': stats_result['slope_stderr'],
            'r_squared': stats_result['r_squared'],
            'p_value': stats_result['p_value'],
            'theoretical_test_pvalue': stats_result['theoretical_test_pvalue']
        })

        all_results.append(pd.DataFrame({'regime': regime_name, 'K': valid_K_values, 'MSE': mse_vs_K}))
        ax.plot(valid_K_values, mse_vs_K, 'o-', label=f'{regime_name} (slope={stats_result["slope"]:.2f})')

    # Add reference slope line for O(K^-1)
    theoretical_slope_K = -1.0
    theoretical_mse_K = (np.array(valid_K_values)**theoretical_slope_K) * highest_initial_error / (valid_K_values[0]**theoretical_slope_K)
    ax.plot(valid_K_values, theoretical_mse_K, '--', color='gray', linewidth=2, label=f'Reference $O(K^{{{theoretical_slope_K:.1f}}})$')

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    pd.concat(all_results, ignore_index=True).to_csv('plots/practical_analysis_data.csv', index=False)
    pd.DataFrame(summary_results).to_csv('plots/practical_analysis_summary.csv', index=False)

    ax.set_xlabel("Number of Blocks $K$", fontsize=14)
    ax.set_ylabel("Monte Carlo MSE", fontsize=14)
    ax.set_xscale('log', base=2)  # Change to base 2 for powers of 2
    ax.set_yscale('log', base=10)
    # Let matplotlib set x-axis limits automatically
    ax.set_title(f"MSE vs. Number of Blocks for a Fixed-Length Path ({total_steps}) ($p \\rightarrow 2^+$)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    plt.savefig("plots/practical_analysis.png", dpi=300)
    plt.close()
    print("Results saved to plots/practical_analysis_data.csv, summary.csv, and .png")