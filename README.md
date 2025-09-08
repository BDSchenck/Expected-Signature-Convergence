# Convergence Theory for Expected Signature Estimation from Dependent Single Paths

**Master's Thesis - ETH Zürich**  
**Author:** Bryson Schenck  
**Advisor:** Prof. Dr. Josef Teichmann  
**Year:** 2025

## Abstract

This thesis develops a convergence theory for empirical expected-signature estimators from single-path data under segment-stationarity (shift-invariant path segments with exponentially decaying serial dependence) and applies it to parameter calibration of two-dimensional Ornstein-Uhlenbeck processes. The main result establishes finite-sample mean-squared-error convergence at rate **O(N^{-2/p})** where N controls block size and p > 2 denotes path regularity. Through systematic hyperparameter optimization, signature methods achieve **10-32% accuracy improvements** over Batched MLE in slow mean-reversion regimes while maintaining **9-15% computational speedups**.

## Key Contributions

### 1. Theoretical Convergence Rate
Proves **O(N^{-2/p})** convergence for block-averaged signature estimators from dependent single paths under three assumptions:
- **(M)** Polynomial moment bounds on p-variation norms
- **(A)** Exponential α-mixing with geometric decay  
- **(S)** Segment-stationarity for mean-reverting processes

### 2. Analytical OU Framework
Reduces expected signature computation to matrix exponentials: **E[S^{(M)}(X)_{0,T}] = exp(T·G^{(M)})·1** where G^{(M)} is the lifted generator on the tensor algebra T^{(M)}(ℝ^d).

### 3. Calibration Methodology
- **MSE + StdDev scoring**: Robust hyperparameter selection with λ = 1.0 
- **K vs K*** distinction: Hyperparameter optimization vs algorithmic selection
- **Rescaled signatures**: Parameter-dependent transport for steeper optimization slopes

## Empirical Results

### Convergence Validation
Log-log MSE-versus-N slopes between **-1.86 and -3.61** observed across parameter regimes, confirming theoretical predictions.

### Calibration Performance
| Parameter Regime | Signature Improvement | Statistical Significance |
|-----------------|----------------------|-------------------------|
| Slow Reversion, Low Vol | 10% | p < 0.001 |
| Slow Reversion, High Vol | 24-32% | p < 0.001 |
| Fast Reversion | Marginal decline | Not significant |

### Computational Efficiency
- **9-15% speedup** in slow reversion regimes despite larger K values
- Superior convergence rates: 94% vs 15% (slow/high volatility regime)
- **O(d^M K)** complexity scaling with M=2 yielding 7-dimensional tensor algebra

## Repository Contents

- **`Schenck-2025-Expected-Signature-Convergence-Theory.pdf`** - Complete thesis (65 pages)

## Mathematical Framework

### Block Sampling Scheme
- **Block length**: Δt_N = δ/N  
- **Number of blocks**: K_N ≍ N^{1+2β} where β = 1/p
- **Estimator**: Ê[S^{(M)}]_N = (1/K_N) Σ Y_k^{(N)}

### Core Innovation: Segment-Stationarity
Enables consistent estimation from time-homogeneous increments without requiring full process stationarity - crucial for mean-reverting financial processes.

### OU Generator Framework
For Stratonovich OU SDE: **dX_t = θ(μ-X_t)dt + σ∘dW_t**

The lifted generator G^{(M)}(θ,μ,σ) acts on T^{(M)}(ℝ^d) and satisfies:
**E_ψ[S^{(M)}(X)_{0,T}] = exp(T·G^{(M)}(ψ))·1**

This enables:
- Machine-precision validation via analytical computation
- Exact gradients through Fréchet differentials  
- Efficient GPU implementation via structured linear algebra

## Technical Implementation

### Three-Phase Experimental Design
1. **Phase 0**: Analytical MLE initialization across K ∈ {1,2,4,8,16,32,64,128,256}
2. **Phase 1**: Hyperparameter discovery (10 MC replications per configuration)  
3. **Phase 2**: Statistical validation (100 MC replications with Wilcoxon signed-rank tests)

### Novel Scoring Rule
**Score = MSE + λ·StdDev** with λ = 1.0 achieved consensus across signature methods and proved robust for λ ∈ {0.5, 1.0, 1.5}.

## Citation

```bibtex
@mastersthesis{Schenck2025Convergence,
  title={Convergence Theory for Expected Signature Estimation from Dependent Single Paths with Applications to Parameter Calibration},
  author={Schenck, Bryson},
  year={2025},
  school={ETH Z{\"u}rich},
  type={Master's thesis},
  advisor={Josef Teichmann}
}
```

## Key Theoretical Result

**Theorem (Main Convergence Result)**: Under Assumptions (M), (A), and (S), the block-based signature estimator satisfies:

**E[||Ê[S^{(M)}]_N - E[S^{(M)}]||²] = O(N^{-2/p})**

where the bias term decays as O(N^{-4β}) and variance dominates at O(N^{-2β}) with β = 1/p.

---

*This thesis establishes the first convergence theory for signature estimation from dependent paths and demonstrates practical advantages in financial parameter calibration, bridging rough path theory with statistical applications.*