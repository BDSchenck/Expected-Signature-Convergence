#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONSISTENT PLOTTING UTILITIES
=============================

This module provides consistent plotting utilities for all experiments
to ensure uniform visual presentation throughout the thesis.

Features:
- Standardized color schemes
- Consistent figure sizes and DPI
- Unified axis formatting
- Common plot types for experiments
- Publication-ready quality
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns


class PlotStyle:
    """Consistent plotting style for thesis figures"""
    
    # Color schemes
    REGIME_COLORS = {
        "Slow Reversion, Low Volatility": "#1f77b4",  # Blue
        "Fast Reversion, Low Volatility": "#ff7f0e",  # Orange
        "Slow Reversion, High Volatility": "#2ca02c",  # Green
        "Fast Reversion, High Volatility": "#d62728",  # Red
    }
    
    METHOD_COLORS = {
        "signature": "#1f77b4",  # Blue
        "mle": "#ff7f0e",  # Orange
        "analytical": "#2ca02c",  # Green
        "empirical": "#d62728",  # Red
    }
    
    # Figure sizes
    SINGLE_FIG_SIZE = (10, 6)
    DOUBLE_FIG_SIZE = (14, 6)
    SQUARE_FIG_SIZE = (8, 8)
    WIDE_FIG_SIZE = (12, 5)
    
    # DPI for saving
    SAVE_DPI = 300
    
    # Font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12
    
    @classmethod
    def setup_style(cls):
        """Setup matplotlib style for consistent plotting"""
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Update rcParams for consistent appearance
        mpl.rcParams.update({
            'figure.figsize': cls.SINGLE_FIG_SIZE,
            'figure.dpi': 100,
            'savefig.dpi': cls.SAVE_DPI,
            'font.size': cls.TICK_SIZE,
            'axes.titlesize': cls.TITLE_SIZE,
            'axes.labelsize': cls.LABEL_SIZE,
            'xtick.labelsize': cls.TICK_SIZE,
            'ytick.labelsize': cls.TICK_SIZE,
            'legend.fontsize': cls.LEGEND_SIZE,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Set color palette
        sns.set_palette("husl")
    
    @classmethod
    def get_regime_color(cls, regime_name: str) -> str:
        """Get consistent color for a parameter regime"""
        return cls.REGIME_COLORS.get(regime_name, "#333333")
    
    @classmethod
    def get_method_color(cls, method_name: str) -> str:
        """Get consistent color for a method"""
        return cls.METHOD_COLORS.get(method_name.lower(), "#333333")


def plot_convergence_analysis(
    N_values: List[float],
    mse_values: Dict[str, List[float]],
    title: str = "Convergence Analysis",
    xlabel: str = "N",
    ylabel: str = "MSE",
    theoretical_slope: float = -1.0,
    save_path: Optional[str] = None,
    show_confidence: bool = True
) -> plt.Figure:
    """
    Create standardized convergence analysis plot
    
    Args:
        N_values: List of N values (x-axis)
        mse_values: Dictionary mapping regime names to MSE values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theoretical_slope: Theoretical convergence slope
        save_path: Path to save figure
        show_confidence: Whether to show confidence intervals
        
    Returns:
        Matplotlib figure
    """
    ThesisPlotStyle.setup_style()
    
    fig, ax = plt.subplots(figsize=ThesisPlotStyle.SINGLE_FIG_SIZE)
    
    # Find highest initial error for theoretical line
    highest_initial_error = max(mse_list[0] for mse_list in mse_values.values())
    
    # Plot each regime
    for regime_name, mse_list in mse_values.items():
        color = ThesisPlotStyle.get_regime_color(regime_name)
        ax.plot(N_values, mse_list, 'o-', color=color, label=regime_name, 
                linewidth=2, markersize=8)
        
        # Add confidence intervals if requested
        if show_confidence and len(mse_list) > 3:
            # Simple confidence band (could be enhanced with actual CI calculation)
            mse_array = np.array(mse_list)
            ci_width = 0.1 * mse_array  # 10% CI for illustration
            ax.fill_between(N_values, mse_array - ci_width, mse_array + ci_width,
                          color=color, alpha=0.2)
    
    # Add theoretical slope line
    theoretical_mse = (np.array(N_values)**theoretical_slope) * \
                      highest_initial_error / (N_values[0]**theoretical_slope)
    ax.plot(N_values, theoretical_mse, 'k--', linewidth=2, 
            label=f'Theoretical $O(N^{{{theoretical_slope:.1f}}})$')
    
    # Format axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    
    # Set x-ticks to actual N values
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=ThesisPlotStyle.SAVE_DPI, bbox_inches='tight')
    
    return fig


def plot_calibration_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create calibration method comparison plot
    
    Args:
        summary_df: Summary results DataFrame
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    ThesisPlotStyle.setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=ThesisPlotStyle.DOUBLE_FIG_SIZE)
    axes = axes.flatten()
    
    # 1. Success rates comparison
    ax = axes[0]
    x = np.arange(len(summary_df))
    width = 0.35
    
    sig_bars = ax.bar(x - width/2, summary_df['signature_success_rate'], 
                      width, label='Signature', 
                      color=ThesisPlotStyle.get_method_color('signature'))
    mle_bars = ax.bar(x + width/2, summary_df['mle_success_rate'], 
                      width, label='MLE',
                      color=ThesisPlotStyle.get_method_color('mle'))
    
    ax.set_xlabel('Parameter Regime')
    ax.set_ylabel('Success Rate')
    ax.set_title('Calibration Success Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['regime'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [sig_bars, mle_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    # 2. Computation time comparison
    ax = axes[1]
    ax.bar(x - width/2, summary_df['signature_time_mean'], 
           width, yerr=summary_df['signature_time_std'],
           label='Signature', color=ThesisPlotStyle.get_method_color('signature'),
           capsize=5)
    ax.bar(x + width/2, summary_df['mle_time_mean'], 
           width, yerr=summary_df['mle_time_std'],
           label='MLE', color=ThesisPlotStyle.get_method_color('mle'),
           capsize=5)
    
    ax.set_xlabel('Parameter Regime')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Average Computation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['regime'], rotation=45, ha='right')
    ax.legend()
    
    # 3. Parameter accuracy (Theta MSE)
    if 'signature_theta_mse' in summary_df.columns:
        ax = axes[2]
        ax.bar(x - width/2, summary_df['signature_theta_mse'], 
               width, label='Signature',
               color=ThesisPlotStyle.get_method_color('signature'))
        ax.bar(x + width/2, summary_df['mle_theta_mse'], 
               width, label='MLE',
               color=ThesisPlotStyle.get_method_color('mle'))
        
        ax.set_xlabel('Parameter Regime')
        ax.set_ylabel('MSE')
        ax.set_title('Theta Parameter MSE')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['regime'], rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
    
    # 4. Speedup ratio
    ax = axes[3]
    if 'time_speedup_ratio' in summary_df.columns:
        speedup = summary_df['mle_time_mean'] / summary_df['signature_time_mean']
    else:
        speedup = summary_df['signature_time_mean'] / summary_df['mle_time_mean']
    
    bars = ax.bar(x, speedup, color='darkgreen')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Parameter Regime')
    ax.set_ylabel('Speedup Ratio')
    ax.set_title('Signature vs MLE Speed Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['regime'], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)
    
    fig.suptitle('Calibration Method Comparison', fontsize=ThesisPlotStyle.TITLE_SIZE + 2)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=ThesisPlotStyle.SAVE_DPI, bbox_inches='tight')
    
    return fig


def plot_sensitivity_analysis(
    param_values: List[float],
    results: Dict[str, List[float]],
    param_name: str,
    metric_name: str = "MSE",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create sensitivity analysis plot
    
    Args:
        param_values: Parameter values (x-axis)
        results: Dictionary mapping regime/method names to results
        param_name: Name of parameter being varied
        metric_name: Name of metric being measured
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    ThesisPlotStyle.setup_style()
    
    fig, ax = plt.subplots(figsize=ThesisPlotStyle.SINGLE_FIG_SIZE)
    
    # Plot each series
    for name, values in results.items():
        if "signature" in name.lower():
            color = ThesisPlotStyle.get_method_color('signature')
        elif "mle" in name.lower():
            color = ThesisPlotStyle.get_method_color('mle')
        else:
            color = ThesisPlotStyle.get_regime_color(name)
        
        ax.plot(param_values, values, 'o-', label=name, color=color,
                linewidth=2, markersize=8)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f'Sensitivity Analysis: {metric_name} vs {param_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    if len(param_values) > 1 and max(param_values) / min(param_values) > 10:
        ax.set_xscale('log')
    
    values_flat = [v for vals in results.values() for v in vals]
    if len(values_flat) > 1 and max(values_flat) / min(values_flat) > 10:
        ax.set_yscale('log')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=ThesisPlotStyle.SAVE_DPI, bbox_inches='tight')
    
    return fig


def create_experiment_summary_figure(
    experiment_name: str,
    results_dict: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure for an experiment
    
    Args:
        experiment_name: Name of the experiment
        results_dict: Dictionary containing experiment results
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    ThesisPlotStyle.setup_style()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Add main title
    fig.suptitle(f'{experiment_name} - Summary Results', 
                 fontsize=ThesisPlotStyle.TITLE_SIZE + 4)
    
    # Create grid of subplots based on available data
    # This is a template that can be customized based on specific experiment needs
    
    # Example layout - customize based on actual results structure
    if 'convergence_data' in results_dict:
        ax1 = plt.subplot(2, 2, 1)
        # Add convergence plot
        
    if 'parameter_errors' in results_dict:
        ax2 = plt.subplot(2, 2, 2)
        # Add parameter error plot
        
    if 'timing_data' in results_dict:
        ax3 = plt.subplot(2, 2, 3)
        # Add timing comparison
        
    if 'summary_statistics' in results_dict:
        ax4 = plt.subplot(2, 2, 4)
        # Add summary statistics table
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=ThesisPlotStyle.SAVE_DPI, bbox_inches='tight')
    
    return fig


# Utility functions for common plot elements

def add_statistical_annotation(ax: plt.Axes, x: float, y: float, 
                             p_value: float, test_name: str = "t-test"):
    """Add statistical significance annotation to plot"""
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."
    
    ax.annotate(f'{sig_text}\n({test_name}: p={p_value:.3f})',
                xy=(x, y), xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='yellow', alpha=0.3))


def format_axis_scientific(ax: plt.Axes, axis: str = 'y', 
                          precision: int = 2):
    """Format axis with scientific notation"""
    if axis == 'y':
        ax.ticklabel_format(style='scientific', axis='y', 
                           scilimits=(0, 0), useMathText=True)
        ax.yaxis.major.formatter._useMathText = True
    else:
        ax.ticklabel_format(style='scientific', axis='x', 
                           scilimits=(0, 0), useMathText=True)
        ax.xaxis.major.formatter._useMathText = True


def add_thesis_watermark(fig: plt.Figure, text: str = "PhD Thesis - Draft"):
    """Add subtle watermark to figure"""
    fig.text(0.5, 0.5, text, transform=fig.transFigure,
             ha='center', va='center', fontsize=40,
             color='gray', alpha=0.1, rotation=45)


if __name__ == "__main__":
    # Example usage
    ThesisPlotStyle.setup_style()
    
    # Example convergence plot
    N_values = [8, 10, 14, 20, 27, 37, 50, 69, 94, 128]  # Base-2 log spaced from 8 to 128
    mse_values = {
        "Slow Reversion, Low Volatility": [1e-2, 5e-3, 3e-3, 2e-3, 1.5e-3, 1e-3, 8e-4, 6e-4, 5e-4, 4e-4],
        "Fast Reversion, High Volatility": [2e-2, 1e-2, 6e-3, 4e-3, 3e-3, 2e-3, 1.6e-3, 1.2e-3, 1e-3, 8e-4],
    }
    
    fig = plot_convergence_analysis(
        N_values, mse_values,
        title="Example Convergence Analysis",
        save_path="plots/example_convergence.png"
    )
    plt.show()