# ============================================================================== 
# == Experiment 1: Direct Theorem Verification ("Longer Path") 
# ============================================================================== 
# 
# PURPOSE: 
# This script implements the "Longer Path" experiment, which is the most 
# direct and faithful numerical verification of the main convergence theorem. 
# 
# METHODOLOGY: 
#   - The independent variable is the theoretical granularity parameter `N`. 
#   - For each `N`, the number of blocks `K_N` is calculated using the 
#     formula from the theorem's proof: K_N ~ N^(1+2*alpha). This is the 
#     optimal number of blocks to average to achieve the fastest convergence. 
#   - This means the total path length is VARIABLE and grows with N and K_N. 
#   - The estimator for a block of duration (delta/N) is compared directly 
#     to the analytical truth for that same short duration. 
# 
# EXPECTED OUTCOME: 
# A monotonically decreasing MSE, demonstrating the convergence rate predicted 
# by the theorem. 
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

def run_theorem_verification_experiment(device):
    print("\n\n===== Starting Experiment 1: Direct Theorem Verification ('Longer Path') ======")
    start_time = time.time()

    # --- Theoretical & Simulation Parameters ---
    d = 2
    M = 4
    p = 2.0 + np.finfo(float).eps
    alpha = 1 / p
    delta = 1.0
    A_K = 1.0
    n_steps_per_block = 10
    n_monte_carlo_sims = 100
    batch_size = 100
    N_values = [8, 16, 32, 64]  # Powers of 2, stopping at 64 for computational feasibility

    regime_configs = {
        "Slow Reversion, Low Volatility": {
            'theta_eigenvalue_range': [0.05, 0.2], 'mu_range': [-1, 1],
            'sigma_diag_range': [0.1, 0.3], 'sigma_corr_level': [0.0, 1.0]  # Randomized correlation
        },
        "Fast Reversion, Low Volatility": {
            'theta_eigenvalue_range': [1.0, 3.0], 'mu_range': [-1, 1],
            'sigma_diag_range': [0.1, 0.3], 'sigma_corr_level': [0.0, 1.0]  # Randomized correlation
        },
        "Slow Reversion, High Volatility": {
            'theta_eigenvalue_range': [0.05, 0.2], 'mu_range': [-1, 1],
            'sigma_diag_range': [1.0, 2.0], 'sigma_corr_level': [0.0, 1.0]  # Randomized correlation
        },
        "Fast Reversion, High Volatility": {
            'theta_eigenvalue_range': [1.0, 3.0], 'mu_range': [-1, 1],
            'sigma_diag_range': [1.0, 2.0], 'sigma_corr_level': [0.0, 1.0]  # Randomized correlation
        }
    }

    all_results = []
    summary_results = []
    fig, ax = plt.subplots(figsize=(14, 8))
    highest_initial_error = 0
    
    k_n_values_for_plot = [int(np.ceil(A_K * n**(1 + 2 * alpha))) for n in N_values]

    for regime_name, config in regime_configs.items():
        print(f"\n--- Running Regime: {regime_name} ---")
        mse_vs_N = []
        
        for i, N_param in enumerate(N_values):
            K_num_blocks = k_n_values_for_plot[i]
            block_duration = delta / N_param
            total_duration = K_num_blocks * block_duration
            total_steps = K_num_blocks * n_steps_per_block

            print(f"  N={N_param}, K_N={K_num_blocks}, sims={n_monte_carlo_sims}...")
            
            mse_accumulator = 0.0
            # Process in batches
            n_batches = (n_monte_carlo_sims + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_monte_carlo_sims)
                batch_sims = batch_end - batch_start
                
                # Generate parameters for this batch
                thetas, mus, sigmas = generate_ou_parameters_for_regime(d, device, config, batch_sims)
                
                # Compute analytical signatures on CPU for memory efficiency
                true_sigs = compute_analytical_ou_expected_signature(
                    block_duration, d, M, thetas.cpu(), mus.cpu(), sigmas.cpu(), False
                ).to(device)
                
                # Generate paths for this batch
                paths = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, False)
                
                # Compute MSE for this batch
                mse_batch = batched_calculate_signature_estimator(paths, M, K_num_blocks, true_sigs).item()
                mse_accumulator += mse_batch * batch_sims
                
                # Clear GPU memory after batch
                del paths, thetas, mus, sigmas, true_sigs
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Average MSE across all simulations
            mse = mse_accumulator / n_monte_carlo_sims
            mse_vs_N.append(mse)
        
        if mse_vs_N[0] > highest_initial_error:
            highest_initial_error = mse_vs_N[0]

        # Compute slope with confidence intervals and statistical tests
        stats_result = compute_slope_with_confidence_interval(np.array(N_values), np.array(mse_vs_N))
        
        print(f"    {format_slope_result(stats_result)}")
        
        summary_results.append({
            'regime': regime_name, 
            'empirical_slope_vs_N': stats_result['slope'],
            'slope_ci_lower': stats_result['slope_ci_lower'],
            'slope_ci_upper': stats_result['slope_ci_upper'],
            'slope_stderr': stats_result['slope_stderr'],
            'r_squared': stats_result['r_squared'],
            'p_value': stats_result['p_value'],
            'theoretical_test_pvalue': stats_result['theoretical_test_pvalue']
        })

        all_results.append(pd.DataFrame({'regime': regime_name, 'N': N_values, 'K': k_n_values_for_plot, 'MSE': mse_vs_N}))
        ax.plot(N_values, mse_vs_N, 'o-', label=f'{regime_name} (slope vs N={stats_result["slope"]:.2f})')

    # Add theoretical slope line for O(N^-1)
    theoretical_slope_N = -1.0
    theoretical_mse_N = (np.array(N_values)**theoretical_slope_N) * highest_initial_error / (N_values[0]**theoretical_slope_N)
    ax.plot(N_values, theoretical_mse_N, 'k--', linewidth=2, label=f'Theoretical $O(N^{{{theoretical_slope_N:.1f}}})$')

    # Add empirical slope line for O(K_N^-1)
    theoretical_slope_K = -1.0
    theoretical_mse_K = (np.array(k_n_values_for_plot)**theoretical_slope_K) * highest_initial_error / (k_n_values_for_plot[0]**theoretical_slope_K)
    ax.plot(N_values, theoretical_mse_K, '--', color='gray', linewidth=2, label=f'Reference $O(K^{{{theoretical_slope_K:.1f}}})$')


    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    pd.concat(all_results, ignore_index=True).to_csv('plots/theorem_verification_data.csv', index=False)
    pd.DataFrame(summary_results).to_csv('plots/theorem_verification_summary.csv', index=False)
    
    ax.set_xlabel("Theoretical Granularity Parameter $N$", fontsize=14)
    ax.set_ylabel("Monte Carlo MSE", fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_title("Convergence Rate Analysis: MSE vs. $N$ with Theoretical Bound ($p \\rightarrow 2^+$)", fontsize=16)
    
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    
    # Create secondary x-axis with proper K labels using twiny
    ax2 = ax.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([str(K) for K in k_n_values_for_plot])
    ax2.set_xlabel("Number of Blocks ($K_N$)", fontsize=14)
    ax2.tick_params(axis='x', which='major', labelsize=12)

    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--")
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    plt.savefig("plots/theorem_verification.png", dpi=300)
    plt.close()
    
    # Final memory cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("Results saved to plots/theorem_verification_data.csv, summary.csv, and .png")