# ==============================================================================
# == Experiment 1: Sanity Check with Full Range Expected Signature
# ==============================================================================
#
# PURPOSE:
# This script validates the generator-based analytical framework and confirms that
# all signature computation methods converge correctly across the entire parameter
# space. This serves as the foundational validation for all subsequent tests.
#
# METHODOLOGY:
#   - Uses "Full Range Expected Signature" approach - parameters sampled uniformly
#     across all regimes to validate convergence across the entire parameter space
#   - Compares four different estimator formulations on the same plot
#   - The primary x-axis is the theoretical granularity parameter `N`
#   - A secondary x-axis shows the corresponding number of blocks `K_N`
#   - Uses 1000 Monte Carlo simulations for high statistical power
#
# ==============================================================================

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from .paths import generate_ou_process
from .estimators import batched_calculate_signature_estimator
from ..analytical.expected_signature import compute_analytical_ou_expected_signature, construct_generator_matrix
from .parameters import generate_ou_parameters_for_regime
from .statistical_analysis import compute_slope_with_confidence_interval, format_slope_result

def run_sanity_check_experiment(device):
    print("\n\n===== Starting Experiment 1: Sanity Check with Full Range Validation ======")
    print("This foundational test validates all computational methods across the full parameter space.")
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
    N_values = [8, 16, 32, 64]  # Powers of 2, but stop at 64 for feasibility

    # Full range configuration - encompasses all parameter regimes
    config = {
        'theta_eigenvalue_range': [0.05, 3.0],  # Full range from slow to fast reversion
        'mu_range': [-1, 1],
        'sigma_diag_range': [0.1, 2.0],  # Full range from low to high volatility
        'sigma_corr_level': [0.0, 1.0]  # Full range of correlations
    }

    # Use a list of dictionaries to store batch results for intelligent aggregation
    batch_results_list = []
    
    k_n_values_for_plot = [int(np.ceil(A_K * n**(1 + 2 * alpha))) for n in N_values]

    for i, N_param in enumerate(N_values):
        K_num_blocks = k_n_values_for_plot[i]
        block_duration = delta / N_param
        total_duration = K_num_blocks * block_duration
        total_steps = K_num_blocks * n_steps_per_block

        print(f"  N={N_param}, K_N={K_num_blocks}, sims={n_monte_carlo_sims}...")
        
        # Process in batches
        n_batches = (n_monte_carlo_sims + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_monte_carlo_sims)
            batch_sims = batch_end - batch_start
            
            thetas, mus, sigmas = generate_ou_parameters_for_regime(d, device, config, batch_sims)

            # --- Case 1 & 3: Standard X_t ---
            paths = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, False)
            true_sigs_t = compute_analytical_ou_expected_signature(block_duration, d, M, thetas, mus, sigmas, False)
            estimator_t = batched_calculate_signature_estimator(paths, M, K_num_blocks, true_sigs_t, return_estimator=True)
            mse_standard = torch.mean(torch.sum((estimator_t - true_sigs_t)**2, dim=1)).item()

            # --- Case 2 & 4: Time-Augmented (t, X_t) ---
            paths_aug = generate_ou_process(batch_sims, total_steps, total_duration, d, thetas, mus, sigmas, device, True)
            true_sigs_aug_t = compute_analytical_ou_expected_signature(block_duration, d, M, thetas, mus, sigmas, True)
            estimator_aug_t = batched_calculate_signature_estimator(paths_aug, M, K_num_blocks, true_sigs_aug_t, return_estimator=True)
            mse_time_aug = torch.mean(torch.sum((estimator_aug_t - true_sigs_aug_t)**2, dim=1)).item()

            # --- Rescaling Calculations ---
            remaining_time = 1.0 - block_duration

            # --- Case 3: Rescaled X_t ---
            true_sigs_T = compute_analytical_ou_expected_signature(1.0, d, M, thetas, mus, sigmas, False)
            L_batch = construct_generator_matrix(d, M, thetas, mus, sigmas, False)
            scaling_operators = torch.linalg.matrix_exp(remaining_time * L_batch)
            scaled_estimators = torch.bmm(scaling_operators, estimator_t.unsqueeze(2)).squeeze(2)
            mse_rescaled = torch.mean(torch.sum((scaled_estimators - true_sigs_T)**2, dim=1)).item()

            # --- Case 4: Rescaled (t, X_t) ---
            true_sigs_aug_T = compute_analytical_ou_expected_signature(1.0, d, M, thetas, mus, sigmas, True)
            L_batch_aug = construct_generator_matrix(d, M, thetas, mus, sigmas, True)
            scaling_operators_aug = torch.linalg.matrix_exp(remaining_time * L_batch_aug)
            scaled_estimators_aug = torch.bmm(scaling_operators_aug, estimator_aug_t.unsqueeze(2)).squeeze(2)
            mse_rescaled_aug = torch.mean(torch.sum((scaled_estimators_aug - true_sigs_aug_T)**2, dim=1)).item()
            
            # Append batch results to the list
            batch_results_list.append({
                'N': N_param,
                'K': K_num_blocks,
                'batch_sims': batch_sims,
                'Standard_MSE': mse_standard,
                'Time_Augmented_MSE': mse_time_aug,
                'Rescaled_Standard_MSE': mse_rescaled,
                'Rescaled_Time_Augmented_MSE': mse_rescaled_aug
            })
            
            # Clean up GPU memory
            del paths, paths_aug, thetas, mus, sigmas
            del estimator_t, estimator_aug_t, true_sigs_t, true_sigs_aug_t
            del L_batch, L_batch_aug, scaling_operators, scaling_operators_aug
            del scaled_estimators, scaled_estimators_aug
            torch.cuda.empty_cache() if device == 'cuda' else None

    # --- Intelligent Aggregation using Pandas ---
    results_df = pd.DataFrame(batch_results_list)
    
    # Calculate weighted average MSE for each N
    def weighted_avg(group):
        return np.average(group, weights=results_df.loc[group.index, 'batch_sims'])

    final_mse = results_df.groupby('N').agg({
        'Standard_MSE': weighted_avg,
        'Time_Augmented_MSE': weighted_avg,
        'Rescaled_Standard_MSE': weighted_avg,
        'Rescaled_Time_Augmented_MSE': weighted_avg
    }).reset_index()

    # Store detailed data
    all_results = pd.DataFrame({
        'N': final_mse['N'],
        'K': k_n_values_for_plot,
        'Standard_MSE': final_mse['Standard_MSE'],
        'Time_Augmented_MSE': final_mse['Time_Augmented_MSE'],
        'Rescaled_Standard_MSE': final_mse['Rescaled_Standard_MSE'],
        'Rescaled_Time_Augmented_MSE': final_mse['Rescaled_Time_Augmented_MSE']
    })
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate slopes with confidence intervals and statistical tests
    stats_std = compute_slope_with_confidence_interval(np.array(all_results['N']), np.array(all_results['Standard_MSE']))
    stats_aug = compute_slope_with_confidence_interval(np.array(all_results['N']), np.array(all_results['Time_Augmented_MSE']))
    stats_res = compute_slope_with_confidence_interval(np.array(all_results['N']), np.array(all_results['Rescaled_Standard_MSE']))
    stats_res_aug = compute_slope_with_confidence_interval(np.array(all_results['N']), np.array(all_results['Rescaled_Time_Augmented_MSE']))
    
    print(f"Standard OU Signature: {format_slope_result(stats_std)}")
    print(f"Time-Augmented Signature: {format_slope_result(stats_aug)}")
    print(f"Rescaled Standard Signature: {format_slope_result(stats_res)}")
    print(f"Rescaled Time-Augmented: {format_slope_result(stats_res_aug)}")
    
    # Store results for CSV export
    summary_results = [
        {'estimator': 'Standard', 'slope': stats_std['slope'], 'slope_ci_lower': stats_std['slope_ci_lower'], 
         'slope_ci_upper': stats_std['slope_ci_upper'], 'slope_stderr': stats_std['slope_stderr'],
         'r_squared': stats_std['r_squared'], 'p_value': stats_std['p_value'], 
         'theoretical_test_pvalue': stats_std['theoretical_test_pvalue']},
        {'estimator': 'TimeAug', 'slope': stats_aug['slope'], 'slope_ci_lower': stats_aug['slope_ci_lower'],
         'slope_ci_upper': stats_aug['slope_ci_upper'], 'slope_stderr': stats_aug['slope_stderr'],
         'r_squared': stats_aug['r_squared'], 'p_value': stats_aug['p_value'],
         'theoretical_test_pvalue': stats_aug['theoretical_test_pvalue']},
        {'estimator': 'Rescaled', 'slope': stats_res['slope'], 'slope_ci_lower': stats_res['slope_ci_lower'],
         'slope_ci_upper': stats_res['slope_ci_upper'], 'slope_stderr': stats_res['slope_stderr'],
         'r_squared': stats_res['r_squared'], 'p_value': stats_res['p_value'],
         'theoretical_test_pvalue': stats_res['theoretical_test_pvalue']},
        {'estimator': 'RescaledAug', 'slope': stats_res_aug['slope'], 'slope_ci_lower': stats_res_aug['slope_ci_lower'],
         'slope_ci_upper': stats_res_aug['slope_ci_upper'], 'slope_stderr': stats_res_aug['slope_stderr'],
         'r_squared': stats_res_aug['r_squared'], 'p_value': stats_res_aug['p_value'],
         'theoretical_test_pvalue': stats_res_aug['theoretical_test_pvalue']}
    ]

    ax.plot(all_results['N'], all_results['Standard_MSE'], 's-', color='tab:blue', label=f"Est. for $E[S(X)_t]$ (slope={stats_std['slope']:.2f})")
    ax.plot(all_results['N'], all_results['Time_Augmented_MSE'], 'o-', color='tab:orange', label=f"Est. for $E[S(t,X)_t]$ (slope={stats_aug['slope']:.2f})")
    ax.plot(all_results['N'], all_results['Rescaled_Standard_MSE'], '^-', color='tab:green', label=f"Rescaled Est. for $E[S(X)_T]$ (slope={stats_res['slope']:.2f})")
    ax.plot(all_results['N'], all_results['Rescaled_Time_Augmented_MSE'], 'D-', color='tab:red', label=f"Rescaled Est. for $E[S(t,X)_T]$ (slope={stats_res_aug['slope']:.2f})")

    ax.set_xlabel("Theoretical Granularity Parameter $N$", fontsize=14)
    ax.set_ylabel("Monte Carlo MSE", fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_title("Comparison of Estimator Formulations ($p \\rightarrow 2^+$)", fontsize=16)
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend(fontsize=12)

    # --- Secondary X-axis for K_N ---
    ax2 = ax.twiny()
    ax2.set_xscale('log', base=2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([f'{k}' for k in k_n_values_for_plot])
    ax2.set_xlabel("Number of Blocks ($K=N^2$)", fontsize=14)
    ax2.tick_params(axis='x', which='major', labelsize=12)

    fig.tight_layout()
    
    # Save plot and data
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f'{output_dir}/sanity_check_comparison.png', dpi=300)
    all_results.to_csv(f'{output_dir}/sanity_check_data.csv', index=False)
    pd.DataFrame(summary_results).to_csv(f'{output_dir}/sanity_check_summary.csv', index=False)
    
    print(f"\nPlot saved to {output_dir}/sanity_check_comparison.png")
    print(f"Data saved to {output_dir}/sanity_check_data.csv")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print("===== Experiment 1: Sanity Check Completed ======")
    
    return all_results, summary_results

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_sanity_check_experiment(device)
