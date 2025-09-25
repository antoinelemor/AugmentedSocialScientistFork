#!/usr/bin/env python3
"""
PROJECT:
-------
AugmentedSocialScientistFork

TITLE:
------
demo_interactive.py

MAIN OBJECTIVE:
---------------
This script demonstrates the interactive CLI usage, showing how to quickly
launch the training system without any configuration files or complex setup,
perfect for new users and quick testing.

Dependencies:
-------------
- pathlib (file path handling)
- AugmentedSocialScientistFork (main package)

MAIN FEATURES:
--------------
1) Quick launch of interactive CLI with minimal setup
2) Demonstration of default directory configuration
3) Example of programmatic CLI invocation
4) Shows how to create a simple data file for testing
5) Automated directory creation
6) Interactive mode walkthrough
7) Best practices for first-time users
8) Sample data generation for testing

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path
import json

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from AugmentedSocialScientistFork import TrainingCLI


def create_sample_data():
    """Create a sample data file for demonstration."""

    # Create data directory
    data_dir = Path("./demo_data")
    data_dir.mkdir(exist_ok=True)

    # Create sample JSONL file
    sample_file = data_dir / "sample_training_data.jsonl"

    # Sample multilingual data
    samples = [
        {"text": "This product is amazing!", "label": 1, "lang": "EN"},
        {"text": "Terrible experience, would not recommend.", "label": 0, "lang": "EN"},
        {"text": "Ce produit est fantastique!", "label": 1, "lang": "FR"},
        {"text": "Très déçu de cet achat.", "label": 0, "lang": "FR"},
        {"text": "Great quality and fast delivery.", "label": 1, "lang": "EN"},
        {"text": "Ne fonctionne pas comme prévu.", "label": 0, "lang": "FR"},
        {"text": "Excellent service!", "label": 1, "lang": "EN"},
        {"text": "Produit défectueux.", "label": 0, "lang": "FR"},
    ]

    # Write samples to file
    with open(sample_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ Created sample data file: {sample_file}")
    print(f"   - {len(samples)} samples")
    print(f"   - Languages: EN, FR")
    print(f"   - Binary classification task")

    return data_dir


def main():
    """Demonstrate interactive CLI usage."""

    print("\n" + "="*70)
    print(" "*10 + "🎯 INTERACTIVE CLI DEMONSTRATION")
    print(" "*15 + "Quick Start Guide")
    print("="*70)

    print("\nThis demo shows how to use the interactive CLI for training.")
    print("\n1️⃣  First, let's create some sample data...")

    # Create sample data
    data_dir = create_sample_data()

    print("\n2️⃣  Now, let's launch the interactive CLI...")
    print("\n" + "-"*50)

    # Setup directories
    models_dir = Path("./demo_models")
    logs_dir = Path("./demo_logs")

    # Create CLI instance
    cli = TrainingCLI(
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        verbose=True
    )

    print("\n📁 Configured directories:")
    print(f"   Data: {data_dir}")
    print(f"   Models: {models_dir}")
    print(f"   Logs: {logs_dir}")

    print("\n3️⃣  Launching interactive mode...")
    print("   You will be presented with a menu to:")
    print("   - Select training mode (multi-label, benchmark, or legacy)")
    print("   - Choose data files")
    print("   - Configure training parameters")
    print("   - Select models (or use auto-selection)")

    print("\n" + "="*70)
    print("💡 TIP: The system will automatically detect that you have")
    print("   both English and French data and suggest appropriate models!")
    print("="*70)

    input("\n⏎ Press Enter to start the interactive CLI...")

    # Run the interactive CLI
    try:
        exit_code = cli.run()

        if exit_code == 0:
            print("\n✅ Training completed successfully!")
            print(f"\n📊 Results saved in:")
            print(f"   Models: {models_dir}")
            print(f"   Logs: {logs_dir}")
        else:
            print(f"\n⚠️  CLI exited with code: {exit_code}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()