# Expected Signature Convergence

This project provides a theoretical and practical framework for estimating the expected signature of a stochastic process from a single, dependent path, based on the research from the Master's Thesis, "Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration." This directory contains the code to reproduce the experiments and results presented in the thesis.

## Overview

The central goal of this work is to establish a rigorous statistical foundation for signature-based inference in settings where data is serially dependent, a common scenario in financial time series analysis. The project introduces a block-based empirical estimator for the expected signature and proves its convergence, providing a tool for non-parametric analysis of stochastic processes.

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

### Main Script

The primary script for running the experiments is `scripts/run_regime_studies.py`.

### Running All Regimes

To run the full suite of experiments across all four parameter regimes, execute the following command:

```bash
python scripts/run_regime_studies.py --monte-carlo 100
```

This will run 100 Monte Carlo simulations for each of the four regimes.

### Running a Single Regime

You can also run a single regime by specifying the `--regime` argument. The available regimes are:
- `slow_low`: Slow Reversion, Low Volatility
- `fast_low`: Fast Reversion, Low Volatility
- `slow_high`: Slow Reversion, High Volatility
- `fast_high`: Fast Reversion, High Volatility

For example, to run the "Slow Reversion, High Volatility" regime with 50 Monte Carlo simulations:

```bash
python scripts/run_regime_studies.py --regime slow_high --monte-carlo 50
```

### Test Mode

For a quick test of the setup, you can use the `--test` flag. This will run a small number of Monte Carlo simulations (2) to ensure everything is working correctly:

```bash
python scripts/run_regime_studies.py --test
```

### Specifying the Device

You can specify the computation device (e.g., `cuda` or `cpu`) using the `--device` argument:

```bash
python scripts/run_regime_studies.py --device cuda
```

By default, the script will use a CUDA-enabled GPU if available, and fall back to the CPU otherwise.

### Outputs

The script will generate the following outputs:
-   **Logs**: A `regime_studies.log` file with detailed information about the experiment run.
-   **Plots**: PNG files of the optimization landscape and other analysis plots will be saved in a timestamped subdirectory within the `plots` directory.
-   **Results**: Detailed results of the experiments will be saved as CSV files in a `detailed_results` subdirectory within the timestamped plots directory.

## Citation

If you use the work or code from this project, please cite the following thesis:

Schenck, B. D. (2025). *Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration*. Master's Thesis, Department of Mathematics, ETH Zürich.