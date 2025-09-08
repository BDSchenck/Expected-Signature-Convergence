# Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration

This repository contains the complete implementation and experiments for the Master's Thesis by Bryson D. Schenck (ETH Zürich, 2025).

📄 **[Full Thesis PDF (65 pages)](./Schenck-2025-Expected-Signature-Convergence-Theory.pdf)**

## Abstract

This thesis develops a convergence theory for empirical expected-signature estimators from single-path data under a segment-stationarity assumption and applies it to parameter calibration of two-dimensional Ornstein-Uhlenbeck processes. A segmentation and reindexing procedure is introduced that retains mean reversion and serial dependence across blocks without assuming block independence, yielding an estimator with finite-sample mean-squared-error convergence at rate O(N^{-2/p}) under exponential α-mixing. Validation uses a generator-based framework that reduces expected-signature computation in linear SDEs to matrix exponentials, enabling precise numerical verification. For calibration, signature-based methods achieve 10-32% improvement over Batched MLE in slow mean-reversion regimes, with 9-15% computational speedup, establishing clear advantages in both statistical accuracy and computational efficiency.

## Core Theoretical Contribution

The main theoretical result of this research is a finite-sample convergence guarantee for the empirical expected-signature estimator. The theorem establishes a mean-squared error convergence rate of $O(N^{-2/p})$, where $N$ is a granularity parameter controlling the size of the blocks and $p > 2$ characterizes the path's regularity. This result is derived under assumptions of segment-stationarity and exponential $\alpha$-mixing, which are suitable for a wide range of processes, including those with mean-reverting dynamics. The proof architecture relies on a careful bias-variance decomposition, using tools from rough path theory to control the bias and covariance inequalities for Hilbert-space-valued mixing sequences to control the variance.

## Application to Ornstein-Uhlenbeck Processes

The Ornstein-Uhlenbeck (OU) process is a cornerstone of this research, serving as the primary model for validation and application. The OU process, a model for mean-reverting stochastic dynamics, is described by the stochastic differential equation:

$dX_t = \theta(\mu - X_t)dt + \sigma \circ dW_t$

This process is central to the thesis for several reasons.

First, it serves as an ideal test case because it satisfies all the necessary assumptions (moment control, mixing, and segment-stationarity) for the convergence theory to apply.

Second, the linear structure of the OU process allows for the development of a complete analytical framework. A key innovation of this work is the use of a lifted generator on the truncated tensor algebra to compute the exact expected signature of the OU process via a matrix exponential, $\mathbb{E}[S^{(M)}(X)_{0,T}] = \exp(T \cdot \mathcal{G}^{(M)}(\psi)) \mathbf{1}$. This provides a machine-precision ground truth, which is invaluable for the numerical verification of the convergence theory.

Third, the OU model is used extensively in the calibration experiments. The thesis develops a signature-based calibration methodology and compares its performance against a classical Batched Maximum Likelihood Estimation (MLE) approach across four economically motivated parameter regimes of the OU process, spanning different levels of mean-reversion speed and volatility.

## Calibration Methodology and Findings

The project details a full calibration pipeline that leverages the theoretical and analytical results. The calibration is formulated as a deterministic optimization problem, minimizing the distance between the empirical block-averaged signature and the exact model-implied signature.

The research demonstrates that in slow mean-reversion regimes, where path-level features are most informative, the signature-based calibration method achieves a 10-32% improvement in accuracy over Batched MLE. These gains are statistically significant and are accompanied by a 9-15% computational speedup, as the richer feature set leads to more reliable and faster convergence in the optimization process.

A novel scoring function, combining Mean Squared Error (MSE) with the standard deviation of the estimates, was introduced to ensure robust hyperparameter selection, a crucial aspect when dealing with the high variance of estimates in limited Monte Carlo simulations.

## Hardware Requirements

To run the experiments and reproduce the results from the thesis, a modern Nvidia GPU with at least 8GB of memory is **essential**. The computations, particularly for higher-order signatures and large numbers of Monte Carlo simulations, are computationally intensive. GPU acceleration reduces the computation time from years to hours, making the experiments feasible to run.

## Running the Experiments

### Setup

1.  **Create a Python virtual environment.**
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create the output directory:**
    ```bash
    mkdir plots
    ```

## Experiment Types

This project contains two main types of experiments:

### 1. Foundational Numerical Experiments

These experiments verify the core convergence theory and include:
- Theorem verification
- Practical analysis 
- Sanity checks
- Sensitivity analysis

**To run all foundational experiments:**
```bash
python -m src.numerical.run_foundational_experiments
```

**To run a specific foundational experiment:**
```bash
python -m src.numerical.run_foundational_experiments --experiment theorem_verification
```

Available experiment names:
- `theorem_verification`
- `practical_analysis` 
- `sanity_check`
- `sensitivity_steps`
- `sensitivity_m`

The Monte Carlo simulation count is configured in `config/experiments.yaml`.

### 2. Calibration Experiments

These experiments compare signature-based calibration against MLE across parameter regimes. The calibration system consists of two main phases:

#### Phase 1: Hyperparameter Tuning
Tests different learning rates and K values (number of blocks) across all calibration methods to find optimal parameters.

**Hyperparameter tuning (10 Monte Carlo runs, 5 workers):**
```bash
python scripts/run_regime_studies.py --monte-carlo 10
```

**Quick test for setup verification (2 Monte Carlo runs, 2 workers):**
```bash
python scripts/run_regime_studies.py --test
```

The hyperparameter tuning explores the optimization landscape across all 4 parameter regimes to identify the best learning rates and K values for each calibration method.

#### Phase 2: Validation Experiments
Uses the optimal parameters from tuning to run larger-scale validation experiments with statistical significance.

**Full validation as used in thesis (100 Monte Carlo runs, 5 workers):**
```bash
python scripts/run_regime_studies.py --monte-carlo 100
```

**Single regime validation:**
```bash
python scripts/run_regime_studies.py --regime slow_high --monte-carlo 100
```

Available regimes:
- `slow_low`: Slow Reversion, Low Volatility
- `fast_low`: Fast Reversion, Low Volatility  
- `slow_high`: Slow Reversion, High Volatility
- `fast_high`: Fast Reversion, High Volatility

#### Calibration Methods Compared
- **Enhanced MLE**: Improved maximum likelihood estimation with smart initialization
- **Expected Signature**: Direct signature-based calibration using expected signatures
- **Rescaled Signature**: Signature-based calibration with rescaling for improved numerical stability

### Device Selection

Both experiment types will automatically use CUDA if available, or fall back to CPU. The foundational experiments detect the device automatically, while calibration experiments accept a `--device` argument.

### Outputs

#### Foundational Experiments Output
- **Plots**: Convergence plots and analysis figures saved to `plots/` directory
- **Data**: Experimental data saved as CSV files for further analysis
- **Console**: Real-time progress and statistical results

#### Calibration Experiments Output
- **Logs**: `regime_studies.log` file with detailed experiment information
- **Timestamped Results**: Each run creates a timestamped folder in `plots/regime_studies_YYYYMMDD_HHMMSS/` containing:
  - **Optimization Landscapes**: PNG files showing MSE surfaces across different K values for each calibration method
  - **Detailed Results**: CSV files in `detailed_results/` subdirectory with all raw experimental data
  - **Experiment Parameters**: JSON file recording all configuration settings used
- **Real-time Progress**: Console output showing convergence progress, timing, and enhancement rates

## Citation

If you use the work or code from this project, please cite the following thesis:

Schenck, B. D. (2025). *Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration*. Master's Thesis, Department of Mathematics, ETH Zürich.