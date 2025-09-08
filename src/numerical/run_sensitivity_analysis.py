# ==============================================================================
# == Sensitivity Analysis Experiments
# ==============================================================================
#
# PURPOSE:
# This module implements two sensitivity analysis experiments:
# 1. Sensitivity to signature truncation level M
# 2. Sensitivity to path discretization (steps per block)
#
# ==============================================================================

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from .paths import generate_ou_process
from .estimators import batched_calculate_signature_estimator
from ..analytical.expected_signature import compute_analytical_ou_expected_signature, construct_generator_matrix
from .parameters import generate_ou_parameters_for_regime
from .statistical_analysis import compute_slope_with_confidence_interval, format_slope_result

def run_m_sensitivity_experiment(device):
    """
    Experiment 4: How does signature truncation level affect convergence?
    Tests M in {2, 4, 6, 8} for both standard and rescaled estimators.
    """
    print("\n\n===== Starting Experiment 4: Sensitivity to Truncation Level M ======")
    start_time = time.time()
    
    # Parameters
    d = 2
    M_values = [2, 4, 6, 8]
    p = 2.0 + np.finfo(float).eps
    alpha = 1 / p
    delta = 1.0
    A_K = 1.0
    n_steps_per_block = 10
    n_monte_carlo_sims = 100
    batch_size = 100
    N_values = [8, 16, 32, 64]
    
    # Full range configuration - randomized parameters
    config = {
        'theta_eigenvalue_range': [0.05, 3.0],
        'mu_range': [-1, 1],
        'sigma_diag_range': [0.1, 2.0],
        'sigma_corr_level': [0.0, 1.0]  # Randomized correlation
    }
    
    all_results = []
    summary_results = []
    
    # Create figure with subplots for each M
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    k_n_values_for_plot = [int(np.ceil(A_K * n**(1 + 2 * alpha))) for n in N_values]
    
    for m_idx, M in enumerate(M_values):
        print(f"\n--- Testing M = {M} ---")
        
        mse_standard_vs_N = []
        mse_rescaled_vs_N = []
        
        for i, N_param in enumerate(N_values):
            K_num_blocks = k_n_values_for_plot[i]
            block_duration = delta / N_param
            total_duration = K_num_blocks * block_duration
            total_steps = K_num_blocks * n_steps_per_block
            
            print(f"  N={N_param}, K_N={K_num_blocks}, M={M}...")
            
            # Initialize accumulators
            mse_standard_accum = 0.0
            mse_rescaled_accum = 0.0
            
            # Process in batches
            n_batches = (n_monte_carlo_sims + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_monte_carlo_sims)
                batch_sims = batch_end - batch_start
                
                # Generate parameters and paths
                thetas, mus, sigmas = generate_ou_parameters_for_regime(d, device, config, batch_sims)
                paths = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, False)
                
                # Standard estimator
                true_sigs_t = compute_analytical_ou_expected_signature(block_duration, d, M, thetas, mus, sigmas, False)
                estimator_t = batched_calculate_signature_estimator(paths, M, K_num_blocks, true_sigs_t, return_estimator=True)
                mse_standard = torch.mean(torch.sum((estimator_t - true_sigs_t)**2, dim=1)).item()
                mse_standard_accum += mse_standard * batch_sims
                
                # Rescaled estimator
                remaining_time = 1.0 - block_duration
                true_sigs_T = compute_analytical_ou_expected_signature(1.0, d, M, thetas, mus, sigmas, False)
                L_batch = construct_generator_matrix(d, M, thetas, mus, sigmas, False)
                scaling_operators = torch.linalg.matrix_exp(remaining_time * L_batch)
                scaled_estimators = torch.bmm(scaling_operators, estimator_t.unsqueeze(2)).squeeze(2)
                mse_rescaled = torch.mean(torch.sum((scaled_estimators - true_sigs_T)**2, dim=1)).item()
                mse_rescaled_accum += mse_rescaled * batch_sims
                
                # Clean up
                del paths, thetas, mus, sigmas, estimator_t
                del L_batch, scaling_operators, scaled_estimators
                torch.cuda.empty_cache() if device == 'cuda' else None
            
            # Average over all batches
            mse_standard_vs_N.append(mse_standard_accum / n_monte_carlo_sims)
            mse_rescaled_vs_N.append(mse_rescaled_accum / n_monte_carlo_sims)
        
        # Compute slopes
        stats_std = compute_slope_with_confidence_interval(np.array(N_values), np.array(mse_standard_vs_N))
        stats_res = compute_slope_with_confidence_interval(np.array(N_values), np.array(mse_rescaled_vs_N))
        
        print(f"  Standard (M={M}): {format_slope_result(stats_std)}")
        print(f"  Rescaled (M={M}): {format_slope_result(stats_res)}")
        
        # Store results
        summary_results.append({
            'M': M,
            'type': 'Standard',
            'slope': stats_std['slope'],
            'slope_ci_lower': stats_std['slope_ci_lower'],
            'slope_ci_upper': stats_std['slope_ci_upper'],
            'r_squared': stats_std['r_squared']
        })
        summary_results.append({
            'M': M,
            'type': 'Rescaled',
            'slope': stats_res['slope'],
            'slope_ci_lower': stats_res['slope_ci_lower'],
            'slope_ci_upper': stats_res['slope_ci_upper'],
            'r_squared': stats_res['r_squared']
        })
        
        # Plot on subplot
        ax = axes[m_idx]
        color = plt.cm.tab10(m_idx)
        ax.plot(N_values, mse_standard_vs_N, 'o-', color=color, 
                label=f'Standard (slope={stats_std["slope"]:.2f})', linewidth=2)
        ax.plot(N_values, mse_rescaled_vs_N, 'o--', color=color,
                label=f'Rescaled (slope={stats_res["slope"]:.2f})', linewidth=2)
        
        ax.set_xlabel("Theoretical Granularity Parameter $N$", fontsize=12)
        ax.set_ylabel("Monte Carlo MSE", fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.set_title(f"M = {M}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_xticks(N_values)
        ax.set_xticklabels(N_values)
        
        # Store detailed data
        all_results.append(pd.DataFrame({
            'M': M,
            'N': N_values,
            'K': k_n_values_for_plot,
            'Standard_MSE': mse_standard_vs_N,
            'Rescaled_MSE': mse_rescaled_vs_N
        }))
    
    # Create combined plot
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Plot all M values on same plot
    for m_idx, M in enumerate(M_values):
        df = all_results[m_idx]
        color = plt.cm.tab10(m_idx)
        
        # Standard lines (solid)
        ax2.plot(df['N'], df['Standard_MSE'], 'o-', color=color, 
                label=f'M={M} Standard', linewidth=2)
        
    # Then plot all rescaled lines (dotted)
    for m_idx, M in enumerate(M_values):
        df = all_results[m_idx]
        color = plt.cm.tab10(m_idx)
        
        # Rescaled lines (dotted)
        ax2.plot(df['N'], df['Rescaled_MSE'], 'o--', color=color,
                label=f'M={M} Rescaled', linewidth=2)
    
    ax2.set_xlabel("Theoretical Granularity Parameter $N$", fontsize=14)
    ax2.set_ylabel("Monte Carlo MSE", fontsize=14)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.set_title("Sensitivity to Truncation Level $M$: Standard vs Rescaled ($p \\rightarrow 2^+$)", fontsize=16)
    ax2.legend(fontsize=12, ncol=2)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(N_values)
    ax2.set_xticklabels(N_values)
    
    # Secondary x-axis
    ax3 = ax2.twiny()
    ax3.set_xscale('log', base=2)
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xticks(N_values)
    ax3.set_xticklabels([str(K) for K in k_n_values_for_plot])
    ax3.set_xlabel("Number of Blocks $K = N^{2}$", fontsize=14)
    
    # Save results
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    pd.concat(all_results, ignore_index=True).to_csv('plots/sensitivity_m_data.csv', index=False)
    pd.DataFrame(summary_results).to_csv('plots/sensitivity_m_summary.csv', index=False)
    
    # Close the subplot figure without saving
    plt.close(fig)
    
    fig2.tight_layout()
    plt.figure(fig2.number)
    plt.savefig("plots/sensitivity_m.png", dpi=300)
    plt.close('all')
    
    print("Results saved to plots/sensitivity_m_data.csv, summary.csv, and .png")


def run_steps_sensitivity_experiment(device):
    """
    Experiment 5: How sensitive is convergence to path discretization?
    Tests steps_per_block in {1, 10, 100}.
    """
    print("\n\n===== Starting Experiment 5: Sensitivity to Path Discretization ======")
    start_time = time.time()
    
    # Parameters
    d = 2
    M = 4  # Fixed truncation level
    p = 2.0 + np.finfo(float).eps
    alpha = 1 / p
    delta = 1.0
    A_K = 1.0
    steps_per_block_values = [1, 10, 100]
    n_monte_carlo_sims = 100
    batch_size = 100
    N_values = [8, 16, 32, 64]
    
    # Full range configuration
    config = {
        'theta_eigenvalue_range': [0.05, 3.0],
        'mu_range': [-1, 1],
        'sigma_diag_range': [0.1, 2.0],
        'sigma_corr_level': [0.0, 1.0]
    }
    
    all_results = []
    summary_results = []
    fig, ax = plt.subplots(figsize=(14, 8))
    
    k_n_values_for_plot = [int(np.ceil(A_K * n**(1 + 2 * alpha))) for n in N_values]
    
    for steps_idx, n_steps_per_block in enumerate(steps_per_block_values):
        print(f"\n--- Testing steps_per_block = {n_steps_per_block} ---")
        
        mse_vs_N = []
        
        for i, N_param in enumerate(N_values):
            K_num_blocks = k_n_values_for_plot[i]
            block_duration = delta / N_param
            total_duration = K_num_blocks * block_duration
            total_steps = K_num_blocks * n_steps_per_block
            
            print(f"  N={N_param}, K_N={K_num_blocks}, steps/block={n_steps_per_block}...")
            
            # Initialize accumulator
            mse_accum = 0.0
            
            # Process in batches
            n_batches = (n_monte_carlo_sims + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_monte_carlo_sims)
                batch_sims = batch_end - batch_start
                
                # Generate parameters and paths
                thetas, mus, sigmas = generate_ou_parameters_for_regime(d, device, config, batch_sims)
                paths = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, False)
                
                # Compute MSE
                true_sigs = compute_analytical_ou_expected_signature(block_duration, d, M, thetas, mus, sigmas, False)
                mse_batch = batched_calculate_signature_estimator(paths, M, K_num_blocks, true_sigs).item()
                mse_accum += mse_batch * batch_sims
                
                # Clean up
                del paths, thetas, mus, sigmas, true_sigs
                torch.cuda.empty_cache() if device == 'cuda' else None
            
            # Average over all batches
            mse = mse_accum / n_monte_carlo_sims
            mse_vs_N.append(mse)
        
        # Compute slope
        stats_result = compute_slope_with_confidence_interval(np.array(N_values), np.array(mse_vs_N))
        
        print(f"  Steps={n_steps_per_block}: {format_slope_result(stats_result)}")
        
        # Store results
        summary_results.append({
            'steps_per_block': n_steps_per_block,
            'slope': stats_result['slope'],
            'slope_ci_lower': stats_result['slope_ci_lower'],
            'slope_ci_upper': stats_result['slope_ci_upper'],
            'r_squared': stats_result['r_squared']
        })
        
        all_results.append(pd.DataFrame({
            'steps_per_block': n_steps_per_block,
            'N': N_values,
            'K': k_n_values_for_plot,
            'MSE': mse_vs_N
        }))
        
        # Plot
        marker = ['s', 'o', '^'][steps_idx]
        ax.plot(N_values, mse_vs_N, f'{marker}-', 
                label=f'{n_steps_per_block} steps/block (slope={stats_result["slope"]:.2f})',
                linewidth=2, markersize=8)
    
    # Add theoretical slope line
    highest_initial_error = max([df['MSE'].iloc[0] for df in all_results])
    theoretical_slope = -1.0
    theoretical_mse = (np.array(N_values)**theoretical_slope) * highest_initial_error / (N_values[0]**theoretical_slope)
    ax.plot(N_values, theoretical_mse, 'k--', linewidth=2, label=f'Theoretical $O(N^{{{theoretical_slope:.1f}}})$')
    
    ax.set_xlabel("Theoretical Granularity Parameter $N$", fontsize=14)
    ax.set_ylabel("Monte Carlo MSE", fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_title("Sensitivity to Path Discretization ($p \\rightarrow 2^+$)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([str(K) for K in k_n_values_for_plot])
    ax2.set_xlabel("Number of Blocks $K = N^{2}$", fontsize=14)
    
    # Save results
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    pd.concat(all_results, ignore_index=True).to_csv('plots/sensitivity_steps_data.csv', index=False)
    pd.DataFrame(summary_results).to_csv('plots/sensitivity_steps_summary.csv', index=False)
    
    fig.tight_layout()
    plt.savefig("plots/sensitivity_steps.png", dpi=300)
    plt.close()
    
    print("Results saved to plots/sensitivity_steps_data.csv, summary.csv, and .png")


if __name__ == '__main__':
    # Ensure the user specifies which experiment to run
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ['steps', 'm']:
        print("Usage: python -m src.numerical.run_sensitivity_analysis <experiment>")
        print("  <experiment>: 'steps' or 'm'")
        sys.exit(1)
    
    experiment = sys.argv[1]
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run the selected experiment
    if experiment == 'steps':
        run_steps_sensitivity_experiment(device)
    elif experiment == 'm':
        run_m_sensitivity_experiment(device)