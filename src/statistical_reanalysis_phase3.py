"""
Statistical Re-analysis of Phase 3 Validation Results
======================================================
Performs proper statistical testing with Wilcoxon signed-rank test,
win rates, effect sizes, and Holm-Bonferroni correction.

Author: Bryson Schenck
Date: August 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, binomtest
import warnings
warnings.filterwarnings('ignore')

# Load the raw Monte Carlo data
data_path = "plots/validation_studies_20250817_151917/detailed_results/combined_landscape_raw_data.csv"
df = pd.read_csv(data_path)

# Define optimal K values from Phase 1
optimal_k = {
    "Slow Reversion, Low Volatility": {"Expected": 256, "Rescaled": 256},
    "Fast Reversion, Low Volatility": {"Expected": 1024, "Rescaled": 1024},
    "Slow Reversion, High Volatility": {"Expected": 256, "Rescaled": 256},
    "Fast Reversion, High Volatility": {"Expected": 64, "Rescaled": 64}
}

def extract_method_data(df, regime, method_name, k_value=None):
    """Extract MSE values for a specific method and regime."""
    regime_data = df[df['regime'] == regime]
    
    if method_name == "Enhanced_MLE":
        # Enhanced MLE uses consensus K
        method_data = regime_data[
            (regime_data['method'] == method_name) & 
            (regime_data['K'] == 'consensus')
        ]
    else:
        # Signature methods use specific K (convert to string for comparison)
        method_data = regime_data[
            (regime_data['method'] == method_name) & 
            (regime_data['K'] == str(k_value))
        ]
    
    # Sort by monte_carlo_run to ensure alignment
    method_data = method_data.sort_values('monte_carlo_run')
    return method_data['MSE'].values

def hodges_lehmann_estimator(x, y):
    """Compute Hodges-Lehmann estimator (median of pairwise differences)."""
    differences = x - y
    return np.median(differences)

def cohen_d(x, y):
    """Calculate Cohen's d effect size for paired data."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def bootstrap_ci(x, y, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for mean difference."""
    def mean_diff(x_sample, y_sample):
        return np.mean(x_sample - y_sample)
    
    differences = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        diff = mean_diff(x[idx], y[idx])
        differences.append(diff)
    
    alpha = 1 - confidence
    lower = np.percentile(differences, 100 * alpha/2)
    upper = np.percentile(differences, 100 * (1 - alpha/2))
    
    return lower, upper

def holm_bonferroni_correction(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Apply Holm-Bonferroni thresholds
    corrected_p = np.zeros(n)
    for i in range(n):
        threshold = alpha / (n - i)
        if sorted_p[i] <= threshold:
            corrected_p[sorted_indices[i]] = sorted_p[i] * (n - i)
        else:
            # Once we fail to reject, all subsequent hypotheses are not rejected
            corrected_p[sorted_indices[i:]] = 1.0
            break
    
    # Ensure monotonicity
    corrected_p = np.minimum(corrected_p, 1.0)
    
    return corrected_p

# Perform analysis for all regime-method combinations
results = []
all_wilcoxon_p_values = []
comparison_names = []

for regime in optimal_k.keys():
    print(f"\nAnalyzing {regime}...")
    
    # Extract Enhanced MLE data
    mle_data = extract_method_data(df, regime, "Enhanced_MLE")
    
    # Skip if insufficient data
    if len(mle_data) < 2:
        print(f"  Insufficient data for {regime}")
        continue
    
    for sig_method in ["Expected_Signature", "Rescaled_Signature"]:
        # Get optimal K for this method and regime
        k_optimal = optimal_k[regime][sig_method.split('_')[0]]
        
        # Extract signature method data
        sig_data = extract_method_data(df, regime, sig_method, k_optimal)
        
        # Skip if data lengths don't match
        if len(sig_data) != len(mle_data):
            print(f"  Data length mismatch for {sig_method}: {len(sig_data)} vs {len(mle_data)}")
            continue
        
        # Calculate differences (positive = signature better)
        differences = mle_data - sig_data
        
        # 1. Paired t-test (original method)
        t_stat, t_pval = stats.ttest_rel(mle_data, sig_data)
        
        # 2. Wilcoxon signed-rank test
        if np.all(differences == 0):
            wilcox_stat, wilcox_pval = np.nan, 1.0
        else:
            wilcox_stat, wilcox_pval = wilcoxon(differences, alternative='two-sided')
        
        # 3. Win rate analysis
        wins_signature = np.sum(sig_data < mle_data)
        win_rate = wins_signature / len(mle_data)
        binom_pval = binomtest(wins_signature, n=len(mle_data), p=0.5, alternative='two-sided').pvalue
        
        # 4. Effect sizes
        hl_estimator = hodges_lehmann_estimator(mle_data, sig_data)
        cohen_d_value = cohen_d(mle_data, sig_data)
        
        # 5. Bootstrap confidence intervals
        boot_lower, boot_upper = bootstrap_ci(mle_data, sig_data)
        
        # 6. Summary statistics
        mle_mean, mle_std = np.mean(mle_data), np.std(mle_data, ddof=1)
        mle_median, mle_iqr = np.median(mle_data), np.percentile(mle_data, 75) - np.percentile(mle_data, 25)
        
        sig_mean, sig_std = np.mean(sig_data), np.std(sig_data, ddof=1)
        sig_median, sig_iqr = np.median(sig_data), np.percentile(sig_data, 75) - np.percentile(sig_data, 25)
        
        # Calculate percentage improvement
        pct_improvement = (mle_mean - sig_mean) / mle_mean * 100
        
        # Store results
        result = {
            'Regime': regime,
            'Method': sig_method.replace('_', ' '),
            'K': k_optimal,
            'N': len(mle_data),
            # MLE statistics
            'MLE_Mean': mle_mean,
            'MLE_Std': mle_std,
            'MLE_Median': mle_median,
            'MLE_IQR': mle_iqr,
            # Signature statistics
            'Sig_Mean': sig_mean,
            'Sig_Std': sig_std,
            'Sig_Median': sig_median,
            'Sig_IQR': sig_iqr,
            # Comparisons
            'Mean_Improvement_%': pct_improvement,
            'Win_Rate': win_rate,
            # Statistical tests
            'T_Test_p': t_pval,
            'Wilcoxon_p': wilcox_pval,
            'Binomial_p': binom_pval,
            # Effect sizes
            'Hodges_Lehmann': hl_estimator,
            'Cohen_d': cohen_d_value,
            # Bootstrap CI
            'Boot_CI_Lower': boot_lower,
            'Boot_CI_Upper': boot_upper
        }
        
        results.append(result)
        all_wilcoxon_p_values.append(wilcox_pval)
        comparison_names.append(f"{regime} - {sig_method}")

# Apply Holm-Bonferroni correction
corrected_p_values = holm_bonferroni_correction(all_wilcoxon_p_values)

# Add corrected p-values to results
for i, result in enumerate(results):
    result['Wilcoxon_p_corrected'] = corrected_p_values[i]
    result['Significant_Original'] = result['T_Test_p'] < 0.05
    result['Significant_Wilcoxon'] = result['Wilcoxon_p'] < 0.05
    result['Significant_Corrected'] = result['Wilcoxon_p_corrected'] < 0.05

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save detailed results
output_path = "plots/validation_studies_20250817_151917/detailed_results/statistical_reanalysis_complete.csv"
results_df.to_csv(output_path, index=False)

# Print summary
print("\n" + "="*80)
print("STATISTICAL RE-ANALYSIS SUMMARY")
print("="*80)

for _, row in results_df.iterrows():
    print(f"\n{row['Regime']} - {row['Method']} (K={row['K']})")
    print(f"  Mean Improvement: {row['Mean_Improvement_%']:.1f}%")
    print(f"  Win Rate: {row['Win_Rate']:.1%} ({int(row['Win_Rate']*row['N'])}/{row['N']} wins)")
    print(f"  P-values:")
    print(f"    Original t-test: {row['T_Test_p']:.4f}")
    print(f"    Wilcoxon: {row['Wilcoxon_p']:.4f}")
    print(f"    Wilcoxon (corrected): {row['Wilcoxon_p_corrected']:.4f}")
    print(f"  Significance: Original={row['Significant_Original']}, Wilcoxon={row['Significant_Wilcoxon']}, Corrected={row['Significant_Corrected']}")

print(f"\nResults saved to: {output_path}")