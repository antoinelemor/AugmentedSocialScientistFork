#!/usr/bin/env python3
"""
PROJECT:
-------
AugmentedSocialScientistFork

TITLE:
------
train_models.py

MAIN OBJECTIVE:
---------------
This script provides a standalone entry point for the AugmentedSocialScientistFork
training CLI, allowing direct command-line execution without package installation
for training BERT-based models with comprehensive configuration options.

Dependencies:
-------------
- sys & pathlib (Python standard library)
- AugmentedSocialScientistFork.cli (main CLI implementation)

MAIN FEATURES:
--------------
1) Direct command-line execution without package installation
2) Automatic path configuration for package imports
3) Full CLI functionality (interactive and non-interactive modes)
4) Support for all training modes (multi-label, benchmark, legacy)
5) Command-line argument parsing for directories and configuration
6) Can be installed as system command via pip
7) Cross-platform compatibility (Windows, macOS, Linux)
8) Executable script with shebang for Unix-like systems

Author:
-------
Antoine Lemor
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CLI main function
from AugmentedSocialScientistFork.cli import main

def run_cli():
    """
    Run the training CLI with enhanced functionality.

    This function serves as the entry point for the standalone script,
    providing full access to all CLI features including:
    - Interactive mode with guided configuration
    - Multi-label training with automatic model selection
    - Benchmark mode for model comparison
    - Language-aware model selection
    - Comprehensive logging and metrics
    """
    # Check if no arguments provided, launch interactive mode directly
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print(" "*10 + "🚀 AugmentedSocialScientistFork Training CLI")
        print(" "*15 + "Advanced Model Training System")
        print("="*70)
        print("\n💡 Launching interactive mode...")
        print("   (Use --help for command-line options)\n")

        # Create default directories if they don't exist
        default_data_dir = Path("./data")
        default_models_dir = Path("./models")
        default_logs_dir = Path("./logs")

        # Ask user for directories interactively
        print("📁 DIRECTORY CONFIGURATION")
        print("-" * 40)

        # Data directory
        data_input = input(f"Data directory [{default_data_dir}]: ").strip()
        data_dir = Path(data_input) if data_input else default_data_dir

        # Models directory
        models_input = input(f"Models directory [{default_models_dir}]: ").strip()
        models_dir = Path(models_input) if models_input else default_models_dir

        # Logs directory
        logs_input = input(f"Logs directory [{default_logs_dir}]: ").strip()
        logs_dir = Path(logs_input) if logs_input else default_logs_dir

        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n✅ Directories configured:")
        print(f"   Data: {data_dir}")
        print(f"   Models: {models_dir}")
        print(f"   Logs: {logs_dir}")

        # Import and run the CLI
        from AugmentedSocialScientistFork import TrainingCLI

        cli = TrainingCLI(
            data_dir=data_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            verbose=True
        )

        # Run the interactive CLI
        try:
            exit_code = cli.run()
            sys.exit(exit_code)
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    # If arguments are provided, use argparse as before
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_cli()