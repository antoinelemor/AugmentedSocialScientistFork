#!/usr/bin/env python3
"""
PROJECT:
-------
AugmentedSocialScientistFork

TITLE:
------
use_cli_programmatically.py

MAIN OBJECTIVE:
---------------
This script demonstrates programmatic usage of the TrainingCLI, showing how to
integrate the training system into custom scripts with full control over
configuration, model selection, and training parameters without user interaction.

Dependencies:
-------------
- pathlib (file path handling)
- json (configuration files)
- AugmentedSocialScientistFork (main package components)

MAIN FEATURES:
--------------
1) Example 1: Simple interactive training with auto-configuration
2) Example 2: Fully programmatic training without user interaction
3) Example 3: Benchmark mode for model comparison
4) Example 4: Custom model selection for specific languages
5) Example 5: Language-aware training with automatic model selection
6) Complete configuration examples for all training modes
7) Demonstration of parallel training and reinforced learning
8) Best practices for integrating the CLI into production pipelines

Author:
-------
Antoine Lemor
"""

from pathlib import Path
import json
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from AugmentedSocialScientistFork import (
    TrainingCLI,
    TrainingConfig,
    BenchmarkConfig
)


def example_1_simple_training():
    """Example 1: Simple multi-label training with auto-configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Multi-Label Training")
    print("="*60)

    # Setup directories
    data_dir = Path("./data")
    models_dir = Path("./models/example1")
    logs_dir = Path("./logs/example1")

    # Create CLI instance
    cli = TrainingCLI(
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        verbose=True
    )

    # Run interactive mode
    exit_code = cli.run()
    print(f"\nTraining completed with exit code: {exit_code}")


def example_2_programmatic_training():
    """Example 2: Fully programmatic training without user interaction."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Programmatic Training")
    print("="*60)

    from AugmentedSocialScientistFork import MultiLabelTrainer

    # Configure training
    config = TrainingConfig(
        n_epochs=10,
        batch_size=32,
        learning_rate=5e-5,
        auto_select_model=True,  # Auto-select best model
        train_by_language=False,
        multilingual_model=False,
        reinforced_learning=True,
        n_epochs_reinforced=5,
        track_languages=True,
        output_dir="./models/example2",
        parallel_training=False
    )

    # Create trainer
    trainer = MultiLabelTrainer(config)

    # Train on data file
    data_file = Path("./data/training_data.jsonl")

    if data_file.exists():
        models = trainer.train(
            data_file=str(data_file),
            auto_split=True,
            split_ratio=0.8,
            stratified=True,
            output_dir=str(config.output_dir)
        )

        print(f"\n✅ Trained {len(models)} models")
        for name, info in models.items():
            metrics = info.performance_metrics
            print(f"  - {name}: F1={metrics.get('macro_f1', 0):.3f}")
    else:
        print(f"❌ Data file not found: {data_file}")


def example_3_benchmark_only():
    """Example 3: Run benchmark to compare models."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Benchmark Mode")
    print("="*60)

    from AugmentedSocialScientistFork import BenchmarkRunner

    # Configure benchmark
    config = BenchmarkConfig(
        epochs=3,
        batch_size=32,
        learning_rate=5e-5,
        balance_benchmark_classes=False,
        test_split_size=0.2,
        save_benchmark_csv=True,
        track_languages=True
    )

    # Create runner
    runner = BenchmarkRunner(
        data_root=Path("./data"),
        models_root=Path("./models/benchmark"),
        config=config
    )

    # Find a data file
    data_files = list(Path("./data").glob("*.jsonl"))
    if data_files:
        data_file = data_files[0]
        print(f"\n📁 Using data file: {data_file.name}")

        # Run comprehensive benchmark
        best_model = runner.run_comprehensive_benchmark(
            data_path=data_file,
            benchmark_epochs=3,
            test_all_models=True,  # Test all appropriate models
            models_to_test=None,    # Auto-select based on languages
            allow_user_selection=False,  # No user interaction
            verbose=True,
            save_detailed_log=True,
            save_best_models_log=True
        )

        if best_model:
            print(f"\n🏆 Best model: {best_model}")
        else:
            print("\n⚠️  No model selected")
    else:
        print("❌ No data files found")


def example_4_custom_model_selection():
    """Example 4: Training with specific model selection."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Model Selection")
    print("="*60)

    from AugmentedSocialScientistFork import (
        ModelSelector,
        CamembertaV2Base,
        RoBERTaBase
    )

    # Use ModelSelector to choose models
    selector = ModelSelector(verbose=True)

    # Get French models for French data
    french_models = []
    for name, profile in selector.MODEL_PROFILES.items():
        if 'fr' in profile.supported_languages:
            french_models.append(name)

    print(f"\n📌 Available French models: {', '.join(french_models[:5])}")

    # Train with specific model
    config = TrainingConfig(
        model_class=CamembertaV2Base,  # Use CamemBERTa-v2 for French
        auto_select_model=False,
        n_epochs=5,
        batch_size=16,
        output_dir="./models/french_model"
    )

    print(f"\n✅ Configuration set for CamemBERTa-v2 training")


def example_5_language_aware_training():
    """Example 5: Language-aware training with automatic model selection."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Language-Aware Training")
    print("="*60)

    # This example shows how the system automatically selects
    # appropriate models based on detected languages

    config = TrainingConfig(
        auto_select_model=True,
        train_by_language=True,  # Separate models per language
        n_epochs=10,
        output_dir="./models/multilingual"
    )

    print("\n📊 Configuration:")
    print("  - Auto-select models: Yes")
    print("  - Train by language: Yes")
    print("  - Models will be selected based on detected languages")
    print("\nThe system will:")
    print("  1. Detect languages in your data")
    print("  2. Select appropriate models for each language")
    print("  3. Train separate models for each language")
    print("  4. Use multilingual models (mDeBERTa, XLM-RoBERTa) for mixed data")


def main():
    """Run examples based on command-line argument."""

    if len(sys.argv) > 1:
        example = sys.argv[1]

        if example == "1":
            example_1_simple_training()
        elif example == "2":
            example_2_programmatic_training()
        elif example == "3":
            example_3_benchmark_only()
        elif example == "4":
            example_4_custom_model_selection()
        elif example == "5":
            example_5_language_aware_training()
        else:
            print("❌ Unknown example. Use 1-5")
    else:
        print("\n" + "="*70)
        print("AUGMENTEDSOCIALSCIENTISTFORK CLI EXAMPLES")
        print("="*70)
        print("\nUsage: python use_cli_programmatically.py [example_number]")
        print("\nAvailable examples:")
        print("  1 - Simple multi-label training (interactive)")
        print("  2 - Programmatic training (no interaction)")
        print("  3 - Benchmark mode to compare models")
        print("  4 - Custom model selection")
        print("  5 - Language-aware training")
        print("\nExample: python use_cli_programmatically.py 3")


if __name__ == "__main__":
    main()