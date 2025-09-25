# AugmentedSocialScientistFork Examples

This directory contains example scripts demonstrating how to use the AugmentedSocialScientistFork package.

## Available Examples

### 1. `demo_interactive.py`
**Quick interactive demonstration**

```bash
python demo_interactive.py
```

This script:
- Creates sample multilingual data (English & French)
- Launches the interactive CLI
- Guides you through the training process
- Shows how the system auto-detects languages and selects appropriate models

Perfect for first-time users who want to see the system in action.

### 2. `use_cli_programmatically.py`
**Programmatic CLI usage examples**

```bash
# Show all examples
python use_cli_programmatically.py

# Run specific example
python use_cli_programmatically.py 1  # Simple interactive training
python use_cli_programmatically.py 2  # Programmatic training
python use_cli_programmatically.py 3  # Benchmark mode
python use_cli_programmatically.py 4  # Custom model selection
python use_cli_programmatically.py 5  # Language-aware training
```

This script demonstrates:
- How to use the CLI programmatically
- Different configuration options
- Batch processing
- Custom model selection
- Integration into production pipelines

## Quick Start

### Simplest Method - Interactive CLI

```bash
# From the repository root
python scripts/train_models.py
```

This will:
1. Prompt you for directories (or use defaults)
2. Show an interactive menu
3. Guide you through configuration
4. Automatically select the best models for your data

### With Your Own Data

1. Prepare your data in JSONL format:
```json
{"text": "Sample text", "label": 0, "lang": "EN"}
{"text": "Another sample", "label": 1, "lang": "EN"}
```

2. Run the interactive CLI:
```bash
python scripts/train_models.py
```

3. Follow the prompts to:
   - Select your data file
   - Choose training mode
   - Configure parameters
   - Start training

## Data Format Examples

### Simple Binary Classification
```json
{"text": "This is positive", "label": 1}
{"text": "This is negative", "label": 0}
```

### With Language Information
```json
{"text": "English text", "label": 1, "lang": "EN"}
{"text": "Texte en français", "label": 0, "lang": "FR"}
```

### Multi-label Data
```json
{"text": "Product review", "sentiment": "positive", "quality": 1, "recommend": true}
```

### With Metadata
```json
{"text": "Sample", "label": 1, "id": "doc_001", "lang": "EN", "source": "twitter"}
```

## Model Selection

The system automatically selects appropriate models based on your data:

- **English data** → RoBERTa, DeBERTa-v3, ELECTRA
- **French data** → CamemBERTa-v2, FlauBERT, DistilCamemBERT
- **Multilingual** → mDeBERTa-v3, XLM-RoBERTa
- **Mixed languages** → Combination of language-specific models

## Tips

1. **First Time Users**: Run `demo_interactive.py` to see the system in action
2. **Production Use**: Use `use_cli_programmatically.py` examples for automation
3. **Benchmarking**: Use benchmark mode to find the best model for your data
4. **Large Datasets**: Enable parallel training in the configuration
5. **Imbalanced Data**: Enable reinforced learning for better minority class performance

## Support

For more information, see the main [README](../README.md) or visit:
https://github.com/antoinelemor/AugmentedSocialScientistFork