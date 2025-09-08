# ==============================================================================
# == Statistical Analysis Utilities for Numerical Experiments
# ==============================================================================
#
# PURPOSE:
# This module provides statistical analysis functions for computing confidence
# intervals, hypothesis tests, and regression analysis for convergence experiments.
#
# ==============================================================================

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

def compute_slope_with_confidence_interval(x_values: np.ndarray, y_values: np.ndarray, 
                                         confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Compute log-linear regression slope with confidence intervals and statistical tests.
    
    Args:
        x_values: Independent variable values (e.g., N or K values)
        y_values: Dependent variable values (e.g., MSE values)
        confidence_level: Confidence level for intervals (default 0.95 for 95% CI)
    
    Returns:
        Dictionary containing:
        - slope: Estimated slope
        - intercept: Estimated intercept
        - slope_stderr: Standard error of slope
        - slope_ci_lower: Lower bound of confidence interval
        - slope_ci_upper: Upper bound of confidence interval
        - r_squared: Coefficient of determination
        - p_value: P-value for slope significance test
        - t_statistic: T-statistic for slope
        - theoretical_test_pvalue: P-value for H0: slope >= -1.0 vs H1: slope < -1.0
    """
    
    # Convert to log scale for log-linear regression
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    
    # Perform linear regression on log-transformed data
    n = len(log_x)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Calculate fitted values and residuals
    y_pred = slope * log_x + intercept
    residuals = log_y - y_pred
    
    # Calculate regression statistics
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((log_y - np.mean(log_y))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard error of slope
    x_mean = np.mean(log_x)
    sum_x_squared_deviations = np.sum((log_x - x_mean)**2)
    mse = ss_res / (n - 2)  # Mean squared error
    slope_stderr = np.sqrt(mse / sum_x_squared_deviations)
    
    # Calculate t-statistic and p-value for slope significance
    t_statistic = slope / slope_stderr
    degrees_freedom = n - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), degrees_freedom))
    
    # Calculate confidence interval for slope
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
    slope_ci_lower = slope - t_critical * slope_stderr
    slope_ci_upper = slope + t_critical * slope_stderr
    
    # Hypothesis test for theoretical convergence: H0: slope >= -1.0 vs H1: slope < -1.0
    theoretical_slope = -1.0
    t_theoretical = (slope - theoretical_slope) / slope_stderr
    theoretical_test_pvalue = stats.t.cdf(t_theoretical, degrees_freedom)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_stderr': slope_stderr,
        'slope_ci_lower': slope_ci_lower,
        'slope_ci_upper': slope_ci_upper,
        'r_squared': r_squared,
        'p_value': p_value,
        't_statistic': t_statistic,
        'theoretical_test_pvalue': theoretical_test_pvalue,
        'degrees_freedom': degrees_freedom,
        'confidence_level': confidence_level
    }

def format_slope_result(result: Dict[str, Any]) -> str:
    """
    Format slope analysis results for human-readable output.
    
    Args:
        result: Dictionary from compute_slope_with_confidence_interval
    
    Returns:
        Formatted string describing the statistical analysis
    """
    slope = result['slope']
    ci_lower = result['slope_ci_lower']
    ci_upper = result['slope_ci_upper']
    r_squared = result['r_squared']
    theoretical_p = result['theoretical_test_pvalue']
    confidence_level = result['confidence_level']
    
    theoretical_test = "faster than theoretical" if theoretical_p < 0.05 else "consistent with theoretical"
    
    return (f"Slope: {slope:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] "
            f"({confidence_level*100:.0f}% CI), R² = {r_squared:.3f}, "
            f"convergence {theoretical_test} (p = {theoretical_p:.3f})")

def power_analysis_sample_size(effect_size: float = 0.05, alpha: float = 0.05, 
                              power: float = 0.8) -> int:
    """
    Compute required sample size for detecting slope deviations from theoretical prediction.
    
    Args:
        effect_size: Minimum detectable difference from theoretical slope (-1.0)
        alpha: Type I error rate (significance level)
        power: Desired statistical power (1 - Type II error rate)
    
    Returns:
        Minimum required sample size
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Approximate formula for linear regression power analysis
    # This is a simplified approximation - actual power depends on x-value distribution
    n_approx = ((z_alpha + z_beta) / effect_size)**2 * 2
    
    return int(np.ceil(n_approx))