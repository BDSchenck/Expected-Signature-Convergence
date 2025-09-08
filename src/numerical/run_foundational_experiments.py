# ==============================================================================
# == Foundational Experiments Runner
# ==============================================================================
#
# This script serves as the main entry point for running the foundational
# numerical experiments of this project.
#
# It can run all experiments by default, or a specific experiment can be chosen
# by providing a command-line argument.
#
# Usage:
#   - Run all experiments: python -m src.numerical.run_foundational_experiments
#   - Run a specific experiment: python -m src.numerical.run_foundational_experiments --experiment <name>
#
# Available experiment names:
#   - theorem_verification
#   - practical_analysis
#   - sanity_check
#   - sensitivity_steps
#   - sensitivity_m
#
# ==============================================================================

import torch
import argparse
from .run_theorem_verification import run_theorem_verification_experiment
from .run_practical_analysis import run_practical_analysis_experiment
from .run_sanity_check import run_sanity_check_experiment
from .run_sensitivity_analysis import run_steps_sensitivity_experiment, run_m_sensitivity_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run foundational numerical experiments.")
    parser.add_argument('--experiment', type=str, help="The name of the experiment to run.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    experiments = {
        "theorem_verification": run_theorem_verification_experiment,
        "practical_analysis": run_practical_analysis_experiment,
        "sanity_check": run_sanity_check_experiment,
        "sensitivity_steps": run_steps_sensitivity_experiment,
        "sensitivity_m": run_m_sensitivity_experiment
    }

    if args.experiment:
        if args.experiment in experiments:
            experiments[args.experiment](device)
        else:
            print(f"Error: Experiment '{args.experiment}' not found.")
            print("Available experiments are:", list(experiments.keys()))
    else:
        print("Running all foundational experiments...")
        for name, func in experiments.items():
            func(device)