# Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration

**Master's Thesis - ETH Zürich, 2025**  
**Author:** Bryson D. Schenck | **Advisor:** Prof. Dr. Josef Teichmann

This repository contains the complete implementation and experimental validation for a thesis that establishes the first rigorous convergence theory for empirical expected-signature estimators from serially dependent single-path data, with applications to parameter calibration.

**[Full Thesis PDF (65 pages)](./Schenck-2025-Expected-Signature-Convergence-Theory.pdf)**

## Abstract

This thesis develops a convergence theory for empirical expected-signature estimators from single-path data under (single realizations of potentially multi-dimensional stochastic processes) segment-stationarity (shift-invariant path segments with exponentially decaying serial dependence) and applies it to parameter calibration of two-dimensional Ornstein-Uhlenbeck processes. The main result establishes finite-sample mean-squared-error convergence at rate **O(N^{-2/p})** where N controls block size and p > 2 denotes path regularity. Through systematic hyperparameter optimization, signature methods achieve **10-32% accuracy improvements** over Batched MLE in slow mean-reversion regimes while maintaining **9-15% computational speedups**.

## Principal Theoretical Contribution

**Main Theorem**: Under Assumptions (M), (A), and (S), the block-based signature estimator satisfies:

**E[||Ê[S^{(M)}]_N - E[S^{(M)}]||²] = O(N^{-2/p})**

where the bias term decays as O(N^{-4β}) and variance dominates at O(N^{-2β}) with β = 1/p.

### Core Assumptions
- **(M)** Polynomial moment bounds on p-variation norms
- **(A)** Exponential α-mixing with geometric decay  
- **(S)** Segment-stationarity for mean-reverting processes

### Innovation: Segment-Stationarity
The breakthrough lies in the **segment-stationarity assumption**, which enables consistent estimation from time-homogeneous increments without requiring full process stationarity—crucial for mean-reverting financial processes where traditional IID assumptions fail.

## Empirical Performance Results

| Parameter Regime | Signature Improvement | Computational Speedup | Statistical Significance |
|-----------------|----------------------|----------------------|-------------------------|
| Slow Reversion, Low Volatility | **+10%** | **+9%** | p < 0.001 |
| Slow Reversion, High Volatility | **+32%** | **+15%** | p < 0.001 |
| Fast Reversion Regimes | Marginal decline | No speedup | Not significant |

### Convergence Rate Validation
- **Log-log MSE-versus-N slopes**: -1.86 to -3.61 across all parameter regimes
- **Superior convergence rates**: 94% vs 15% success rates (slow/high volatility regime)
- **Theoretical predictions confirmed** across four economically motivated regimes

## Analytical Framework for Ornstein-Uhlenbeck Processes

### Generator-Based Expected Signature Computation
For Ornstein-Uhlenbeck processes satisfying the Stratonovich SDE:
```
dX_t = θ(μ - X_t)dt + σ ∘ dW_t
```

The key innovation reduces expected signature computation to matrix exponentials:

**E[S^{(M)}(X)_{0,T}] = exp(T · G^{(M)}(θ,μ,σ)) · 1**

where G^{(M)} is the lifted generator on the truncated tensor algebra T^{(M)}(ℝ^d).

### Mathematical Advantages
- **Machine-precision validation** of theoretical convergence rates
- **Exact gradients** through Fréchet differentials of the matrix exponential
- **Computational efficiency**: O(d^M K) complexity with M=2 yielding 7-dimensional tensor algebra
- **GPU optimization** via structured linear algebra operations

## Block Sampling Architecture

### Estimator Construction
- **Block length**: Δt_N = δ/N  
- **Number of blocks**: K_N ≍ N^{1+2β} where β = 1/p
- **Block estimator**: Ê[S^{(M)}]_N = (1/K_N) Σ_{k=1}^{K_N} Y_k^{(N)}

### Theoretical Framework
The proof architecture relies on a bias-variance decomposition:
- **Bias control**: Tools from rough path theory for path regularity
- **Variance control**: Covariance inequalities for Hilbert-space-valued mixing sequences
- **Rate optimization**: Balance between bias O(N^{-4β}) and variance O(N^{-2β})

## Applications in Financial Parameter Calibration

### Problem Formulation
Calibration formulated as deterministic optimization:
```
min_{θ,μ,σ} ||S_emp - E[S^{(M)}(X(θ,μ,σ))]||²
```

### Three-Method Comparison
1. **Enhanced MLE**: Batched maximum likelihood with smart initialization
2. **Expected Signature**: Direct signature matching with analytical expected signatures  
3. **Rescaled Signature**: Parameter-dependent transport for improved numerical stability

### Novel Scoring Rule
**Score = MSE + λ · StdDev** with λ = 1.0, designed to achieve consensus between signature methods. By incorporating standard deviation, the scoring rule ensures that optimal learning rates and K values agree between Expected Signature and Rescaled Signature methods. The framework proves robust across λ ∈ {0.5, 1.0, 1.5}.

## Significance for Stochastic Finance

### Theoretical Foundations
- **First convergence theory** for signature-based parameter estimation from dependent single-path data
- **Segment-stationarity framework** applicable to mean-reverting processes in finance
- **Complete mathematical treatment** of bias-variance trade-offs under serial dependence

## Experimental Validation

### Rigorous Statistical Design
**Phase 0**: Analytical OU parameter selection across K ∈ {1,2,4,8,16,32,64,128,256}
**Phase 1**: Hyperparameter optimization (10 Monte Carlo runs per configuration)  
**Phase 2**: Statistical validation (100 Monte Carlo runs with Wilcoxon signed-rank tests)

### Parameter Regime Coverage
Four economically motivated parameter configurations:
- **Slow Reversion, Low Volatility**: θ ∈ [0.05, 0.2], σ ∈ [0.1, 0.3]
- **Fast Reversion, Low Volatility**: θ ∈ [0.5, 2.0], σ ∈ [0.1, 0.3]
- **Slow Reversion, High Volatility**: θ ∈ [0.05, 0.2], σ ∈ [1.0, 2.0]
- **Fast Reversion, High Volatility**: θ ∈ [0.5, 2.0], σ ∈ [1.0, 2.0]

### Key Experimental Findings
- **Consensus K* selection**: MSE + StdDev scoring with λ = 1.0 achieves method consensus
- **Regime-dependent performance**: Signature methods excel in slow mean-reversion environments
- **Computational efficiency**: 9-15% speedup despite larger K values due to superior convergence

## Reproducibility and Implementation

### Hardware Requirements
Modern Nvidia GPU with at least 8GB memory essential for:
- Higher-order signature computations (M=4, d=2 → 31-dimensional tensors)
- Parallel hyperparameter optimization across K values
- GPU acceleration reduces computation time from years to hours

### Software Architecture
```
src/
├── numerical/           # Foundational convergence experiments
├── analytical/          # Generator-based OU signature computation  
├── calibration_mega/    # Three-phase calibration methodology
└── plotting/           # Visualization and result analysis
```

### Experiment Reproduction
```bash
# Foundational convergence validation (100 MC runs, all in parallel)
python -m src.numerical.run_foundational_experiments

# Hyperparameter optimization (10 MC runs, 5 workers)  
python scripts/run_regime_studies.py --monte-carlo 10

# Full statistical validation (100 MC runs, 5 workers)
python scripts/run_regime_studies.py --monte-carlo 100
```

## Citation

```bibtex
@mastersthesis{Schenck2025Convergence,
  title={Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration},
  author={Schenck, Bryson D.},
  year={2025},
  school={ETH Z{\"u}rich},
  type={Master's thesis},
  advisor={Josef Teichmann},
  pages={65}
}
```

---

*This thesis establishes the first convergence theory for signature estimation from dependent single-path data and demonstrates significant performance advantages in financial parameter calibration, bridging stochastic analysis theory with quantitative finance applications.*
