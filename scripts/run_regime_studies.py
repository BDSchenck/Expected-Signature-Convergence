"""
Main runner script for multi-regime parameter studies.

This script orchestrates the complete multi-regime optimization landscape experiments
for the project's calibration system.

Author: Bryson Schenck
Date: January 2025
"""

import torch
import time
import logging
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.calibration_mega.regime_optimization_experiment import (
    RegimeOptimizationExperiment, 
    run_all_regime_experiments
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regime_studies.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for regime studies."""
    
    parser = argparse.ArgumentParser(description='Run multi-regime parameter studies')
    parser.add_argument('--regime', type=str, default='all',
                       choices=['all', 'slow_low', 'fast_low', 'slow_high', 'fast_high'],
                       help='Which regime(s) to run')
    parser.add_argument('--monte-carlo', type=int, default=10,
                       help='Number of Monte Carlo runs per regime (default: 10)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with 2 Monte Carlo runs')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computing device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set Monte Carlo runs
    n_monte_carlo = 2 if args.test else args.monte_carlo
    
    if args.test:
        logger.info("Running in TEST MODE with 2 Monte Carlo runs")
    
    # Map regime names
    regime_map = {
        'slow_low': "Slow Reversion, Low Volatility",
        'fast_low': "Fast Reversion, Low Volatility",
        'slow_high': "Slow Reversion, High Volatility",
        'fast_high': "Fast Reversion, High Volatility"
    }
    
    start_time = time.time()
    
    if args.regime == 'all':
        logger.info(f"Running ALL 4 regimes with {n_monte_carlo} Monte Carlo runs each")
        logger.info(f"Estimated runtime: ~{n_monte_carlo * 0.5:.1f} hours")
        
        # Run all regimes
        results = run_all_regime_experiments(
            n_monte_carlo=n_monte_carlo,
            device=device
        )
        
        logger.info("\n" + "="*80)
        logger.info("SUMMARY OF ALL REGIME EXPERIMENTS")
        logger.info("="*80)
        
        for regime_name, regime_results in results.items():
            logger.info(f"\n{regime_name}:")
            
            # Find best results
            best_mle_mse = float('inf')
            best_mle_k = None
            for K, mse_list in regime_results.quick_mle_mse.items():
                if mse_list:
                    mean_mse = sum(mse_list) / len(mse_list)
                    if mean_mse < best_mle_mse:
                        best_mle_mse = mean_mse
                        best_mle_k = K
            
            best_exp_mse = float('inf')
            best_exp_k = None
            for K, mse_list in regime_results.expected_sig_mse.items():
                if mse_list:
                    mean_mse = sum(mse_list) / len(mse_list)
                    if mean_mse < best_exp_mse:
                        best_exp_mse = mean_mse
                        best_exp_k = K
            
            best_rsc_mse = float('inf')
            best_rsc_k = None
            for K, mse_list in regime_results.rescaled_sig_mse.items():
                if mse_list:
                    mean_mse = sum(mse_list) / len(mse_list)
                    if mean_mse < best_rsc_mse:
                        best_rsc_mse = mean_mse
                        best_rsc_k = K
            
            logger.info(f"  Best MLE: K={best_mle_k}, MSE={best_mle_mse:.4f}")
            logger.info(f"  Best Expected Sig: K={best_exp_k}, MSE={best_exp_mse:.4f}")
            logger.info(f"  Best Rescaled Sig: K={best_rsc_k}, MSE={best_rsc_mse:.4f}")
            
            if best_mle_mse > 0:
                exp_improvement = (1 - best_exp_mse/best_mle_mse) * 100
                rsc_improvement = (1 - best_rsc_mse/best_mle_mse) * 100
                logger.info(f"  Signature improvements over MLE: Expected={exp_improvement:.1f}%, Rescaled={rsc_improvement:.1f}%")
    
    else:
        # Run single regime
        regime_name = regime_map[args.regime]
        logger.info(f"Running single regime: {regime_name} with {n_monte_carlo} Monte Carlo runs")
        
        experiment = RegimeOptimizationExperiment(
            regime_name=regime_name,
            n_monte_carlo=n_monte_carlo,
            device=device
        )
        
        results = experiment.run_regime_experiment()
        experiment.plot_regime_landscape(results)
        experiment.save_regime_results(results)
        
        logger.info(f"\nCompleted {regime_name} experiment")
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info(f"\nTotal runtime: {hours}h {minutes}m {seconds}s")
    logger.info("All experiments completed successfully!")
    
    # Print output locations
    logger.info("\nOutput files saved to:")
    logger.info("  - Plots: plots/regime_studies/")
    logger.info("  - Results: plots/regime_studies/detailed_results/")
    logger.info("  - Log: regime_studies.log")


if __name__ == "__main__":
    main()