#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Enhanced MLE landscape plot for calibration chapter
This script recreates the EXACT first subplot from the original 4-panel figure
Including ALL methods: Analytical OU, Method of Moments, Enhanced MLE, 
Expected Signature, and Rescaled Signature
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up the plot style to match original EXACTLY
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (7, 5.5),  # Slightly smaller to match subplot size
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Read the data
csv_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\learning_rate_analysis_20250816_052909\experiment_results\optimized_decay_0.05_to_0.01\detailed_results\slow_reversion_high_volatility_results.csv"
df = pd.read_csv(csv_path)

# Filter for different phases/methods - using the CORRECT phase names from the data
analytical_ou = df[df['Phase'] == 'Phase0_AnalyticalOU'].copy()
mom_phase0 = df[df['Phase'] == 'Phase0_MethodOfMoments'].copy()
enhanced_mle_phase1 = df[df['Phase'] == 'Phase1_EnhancedMLE_SmartInit'].copy()
enhanced_mle_phase2 = df[df['Phase'] == 'Phase2_EnhancedMLE'].copy()
expected_sig = df[df['Phase'] == 'Phase2_ExpectedSignature_SmartInit'].copy()
rescaled_sig = df[df['Phase'] == 'Phase2_RescaledSignature_SmartInit'].copy()

# Create the figure with EXACT subplot size
fig, ax = plt.subplots(figsize=(7, 5.5))

# EXACT colors from the original plot
color_analytical = '#1f77b4'  # blue
color_mom = '#ff7f0e'  # orange  
color_enhanced = '#2ca02c'  # green
color_expected = '#d62728'  # red
color_rescaled = '#9467bd'  # purple

# Plot Analytical Batched MLE (Phase 0) - baseline
if len(analytical_ou) > 0:
    ax.errorbar(analytical_ou['K'].values, analytical_ou['mean_MSE'].values, 
                yerr=analytical_ou['std_MSE'].values,
                fmt='o-', color=color_analytical, label='Analytical Batched MLE',
                linewidth=1.5, markersize=5, capsize=2, alpha=0.8)

# Plot Method of Moments (Phase 0)
if len(mom_phase0) > 0:
    ax.errorbar(mom_phase0['K'].values, mom_phase0['mean_MSE'].values,
                yerr=mom_phase0['std_MSE'].values,
                fmt='^-', color=color_mom, label='Method of Moments',
                linewidth=1.5, markersize=5, capsize=2, alpha=0.8)

# Plot Enhanced MLE - use Phase2 data (has more complete K values)
if len(enhanced_mle_phase2) > 0:
    ax.errorbar(enhanced_mle_phase2['K'].values, enhanced_mle_phase2['mean_MSE'].values,
                yerr=enhanced_mle_phase2['std_MSE'].values,
                fmt='s-', color=color_enhanced, label='Enhanced MLE',
                linewidth=1.5, markersize=5, capsize=2, alpha=0.8)

# Plot Expected Signature (Phase 2)
if len(expected_sig) > 0:
    ax.errorbar(expected_sig['K'].values, expected_sig['mean_MSE'].values,
                yerr=expected_sig['std_MSE'].values,
                fmt='D-', color=color_expected, label='Expected Signature',
                linewidth=1.5, markersize=5, capsize=2, alpha=0.8)

# Plot Rescaled Signature (Phase 2)
if len(rescaled_sig) > 0:
    ax.errorbar(rescaled_sig['K'].values, rescaled_sig['mean_MSE'].values,
                yerr=rescaled_sig['std_MSE'].values,
                fmt='v-', color=color_rescaled, label='Rescaled Signature',
                linewidth=1.5, markersize=5, capsize=2, alpha=0.8)

# Mark the optimal K for Enhanced MLE with a star
optimal_k = 16  # K* = 16 for Enhanced MLE
if len(analytical_ou[analytical_ou['K'] == optimal_k]) > 0:
    optimal_mse = analytical_ou[analytical_ou['K'] == optimal_k]['mean_MSE'].values[0]
    ax.scatter([optimal_k], [optimal_mse], color='black', s=150, marker='*', 
               zorder=5, edgecolor='black', linewidth=1, label=f'K* = {optimal_k}')

# Set logarithmic scale - EXACTLY as in original
ax.set_xscale('log', base=2)
ax.set_yscale('log')

# Set axis labels and title - EXACT text and size
ax.set_xlabel('Number of Blocks (K)', fontsize=10)
ax.set_ylabel('Mean Squared Error', fontsize=10)
ax.set_title('Enhanced MLE', fontsize=11)  # Simple title like in subplot

# Set x-axis ticks to match original EXACTLY
k_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
ax.set_xticks(k_ticks)
ax.set_xticklabels([str(k) for k in k_ticks])

# Rotate x-axis labels at 45 degrees
plt.xticks(rotation=45, ha='right')

# Set y-axis limits to match original EXACTLY (looking at the plot, it goes from ~0.2 to ~7)
ax.set_ylim([0.2, 7])

# Set y-axis ticks to match original
y_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{y:.1f}' if y < 1 else f'{int(y)}' for y in y_ticks])

# Add grid - matching original style
ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

# Add legend - EXACT position and style as original
ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
          ncol=1, borderpad=0.3, columnspacing=1.0, handlelength=1.5)

# Improve layout
plt.tight_layout()

# Save the figure
output_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\enhanced_mle_landscape.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF for LaTeX
pdf_path = r"C:\Users\Bryson\Documents\Git\Thesis-Paper\plots\enhanced_mle_landscape.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"PDF saved to: {pdf_path}")

plt.show()