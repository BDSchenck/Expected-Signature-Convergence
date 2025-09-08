"""
This module contains functions for plotting the results of the numerical experiments.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_mse(N_values, results, alpha, filename=None):
    """
    Plots the MSE vs. N on a log-log scale for different experiment configurations.
    If a filename is provided, saves the plot to the 'plots/' directory.
    """
    plt.figure(figsize=(14, 10))
    max_start_mse = 0
    
    for label, mse_values in results.items():
        # Filter out any potential non-positive mse values for log plot
        valid_indices = [i for i, v in enumerate(mse_values) if v > 0]
        if not valid_indices:
            print(f"Warning: No positive MSE values for regime '{label}', skipping plot.")
            continue
        
        if mse_values[0] > max_start_mse:
            max_start_mse = mse_values[0]
            
        N_plot = [N_values[i] for i in valid_indices]
        mse_plot = [mse_values[i] for i in valid_indices]

        # Perform a linear regression on the log-log data to find the empirical slope
        log_N = np.log(N_plot)
        log_mse = np.log(mse_plot)
        empirical_slope, _ = np.polyfit(log_N, log_mse, 1)
        
        label_full = f"{label} (Slope: {empirical_slope:.2f})"
        plt.loglog(N_plot, mse_plot, 'o-', label=label_full, alpha=0.8)

    # Plot the theoretical slope for reference, anchored to the highest starting MSE
    if max_start_mse > 0:
        C = max_start_mse / (N_values[0]**(-2 * alpha))
        plt.loglog(N_values, C * (np.array(N_values, dtype=np.float64)**(-2 * alpha)), 'r--', linewidth=2, label=f'Theoretical Slope (-2*alpha = {-2*alpha:.2f})')
    
    plt.xlabel('N (Block Size Parameter)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE Convergence Rate Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    if filename:
        filepath = f"plots/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
        plt.close()
    else:
        plt.show()

def plot_single_run_convergence(running_avg, true_value):
    """
    Plots the convergence of the running average of the estimator.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(running_avg, label='Running Average of Estimator')
    plt.axhline(y=true_value, color='r', linestyle='--', label='True Expected Signature')
    plt.xlabel('Number of Blocks')
    plt.ylabel('Estimator Value')
    plt.title('Convergence of a Single Run')
    plt.legend()
    plt.grid(True)
    plt.show()