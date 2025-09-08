#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXACT copy of the first subplot from the original 4-panel figure
This uses the EXACT code from regime_optimization_experiment.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the data
csv_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\learning_rate_analysis_20250816_052909\experiment_results\optimized_decay_0.05_to_0.01\detailed_results\slow_reversion_high_volatility_results.csv"
df = pd.read_csv(csv_path)

# Create single plot instead of subplot - but same size as original subplot
fig, ax1 = plt.subplots(figsize=(10, 8))  # Half of (20, 16) since it was 2x2

# Prepare Phase 0: Classical method data
analytical_ou = df[df['Phase'] == 'Phase0_AnalyticalOU'].copy()
analytical_ou_k_vals = analytical_ou['K'].values.tolist()
analytical_ou_mse_means = analytical_ou['mean_MSE'].values.tolist()

mom = df[df['Phase'] == 'Phase0_MethodOfMoments'].copy()
mom_k_vals = mom['K'].values.tolist()
mom_mse_means = mom['mean_MSE'].values.tolist()

# Phase 1 data not needed - Quick MLE removed from plot

# Prepare Phase 2: Batched MLE
batched_mle_phase2 = df[df['Phase'] == 'Phase2_EnhancedMLE'].copy()

# Prepare Phase 3: Signature method data  
expected_sig = df[df['Phase'] == 'Phase2_ExpectedSignature_SmartInit'].copy()
exp_k_vals = expected_sig['K'].values.tolist()
exp_mse_means = expected_sig['mean_MSE'].values.tolist()
exp_mse_stds = expected_sig['std_MSE'].values.tolist()

rescaled_sig = df[df['Phase'] == 'Phase2_RescaledSignature_SmartInit'].copy()
rsc_k_vals = rescaled_sig['K'].values.tolist()
rsc_mse_means = rescaled_sig['mean_MSE'].values.tolist()
rsc_mse_stds = rescaled_sig['std_MSE'].values.tolist()

# EXACT PLOTTING CODE FROM ORIGINAL

# Plot Phase 0: Classical methods (new baseline methods)
if analytical_ou_k_vals:
    ax1.plot(analytical_ou_k_vals, analytical_ou_mse_means, 'D--', label='Analytical Batched MLE',
            markersize=5, linewidth=1.5, color='#8B4513', alpha=0.8)

if mom_k_vals:
    ax1.plot(mom_k_vals, mom_mse_means, 'p--', label='Method of Moments',
            markersize=5, linewidth=1.5, color='#4B0082', alpha=0.8)

# Quick MLE removed per user request

# Plot Phase 3: Signature methods
# Assuming n_monte_carlo > 1 since we have std data
if exp_k_vals:
    ax1.errorbar(exp_k_vals, exp_mse_means, yerr=exp_mse_stds,
               fmt='o-', label='Expected Signature', markersize=8,
               linewidth=2, color='#A23B72', capsize=5)

if rsc_k_vals:
    ax1.errorbar(rsc_k_vals, rsc_mse_means, yerr=rsc_mse_stds,
               fmt='^-', label='Rescaled Signature', markersize=8,
               linewidth=2, color='#F18F01', capsize=5)

# Plot Phase 2: Batched MLE as single X mark at consensus K (design choice)
# Get consensus K (most common K from Phase 1)
consensus_k = 16  # From the data, this is the optimal K
if len(batched_mle_phase2) > 0:
    mean_mse = batched_mle_phase2['mean_MSE'].values[0]
    std_mse = batched_mle_phase2['std_MSE'].values[0]
    
    # Since we have multiple samples, plot with error bars
    # Calculate 95% confidence interval
    n_samples = 10  # Approximate from Phase 1
    if n_samples > 1 and std_mse > 0:
        ci = stats.t.interval(0.95, n_samples-1,
                             loc=mean_mse,
                             scale=std_mse/np.sqrt(n_samples))
        ax1.errorbar(consensus_k, mean_mse, yerr=[[mean_mse - ci[0]], [ci[1] - mean_mse]],
                   fmt='X', markersize=12, color='#2E86AB', markeredgewidth=2,
                   capsize=5, label='Batched MLE')
    else:
        ax1.plot(consensus_k, mean_mse, 'X', markersize=12, color='#2E86AB',
               markeredgewidth=2, label='Batched MLE')

# EXACT AXIS SETTINGS FROM ORIGINAL
ax1.set_xlabel('Number of Blocks (K)', fontsize=14)
ax1.set_ylabel('Monte Carlo MSE', fontsize=14)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_title('Calibration Performance vs Block Count K', fontsize=16)  # Descriptive title

# Note removed per user request - Batched MLE shown at K=16
ax1.grid(True, alpha=0.3, linestyle='--', which='both')
ax1.legend(fontsize=12, loc='best')

# Save the figure
plt.tight_layout()
output_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\enhanced_mle_landscape.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF for LaTeX
pdf_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\enhanced_mle_landscape.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF saved to: {pdf_path}")

plt.show()