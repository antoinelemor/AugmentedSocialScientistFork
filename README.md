# AugmentedSocialScientist

fine‑tuning state‑of‑the‑art transformer models for social‑science projects, with comprehensive tracking, intelligent model selection, multilingual support, and adaptive training strategies.

## 1. overview

this repository provides a comprehensive framework for fine‑tuning transformer models with advanced capabilities for social science research. the package supports both classic BERT variants and cutting‑edge models like DeBERTa‑v3, RoBERTa, ELECTRA, ALBERT, and specialized long‑context models.

## 2. key capabilities

| capability | description |
|------------|-------------|
| **state‑of‑the‑art models** | includes 30+ SOTA models: DeBERTa‑v3 (all sizes), RoBERTa, ELECTRA, ALBERT, BigBird, Longformer, plus specialized multilingual models like mDeBERTa and XLM‑RoBERTa. French models include CamemBERTa‑v2, FlauBERT, DistilCamemBERT, FrALBERT, FrELECTRA. |
| **command‑line interface** | comprehensive CLI for training orchestration with interactive menus, automatic model selection, benchmark mode, and full parameter configuration. supports both interactive and programmatic usage. |
| **language‑aware selection** | automatically detects languages in data and selects appropriate models: French models for French text, English models for English, multilingual models for mixed data. each model tested only on supported languages. |
| **intelligent model selection** | automatic model recommendation based on task complexity, resource constraints, and performance requirements through sophisticated scoring algorithms. |
| **comprehensive benchmarking** | built‑in benchmark runner tests all appropriate models, generates detailed CSV/JSON logs with HuggingFace model names, supports language‑specific evaluation. |
| **multilingual optimization** | dedicated multilingual model selector with language distribution analysis, performance benchmarking per language, and ensemble support for robust cross‑lingual tasks. |
| **metadata‑aware training** | support for JSONL/CSV data with metadata (id, language, custom fields), enabling stratified analysis and per‑language performance tracking throughout training. |
| **multi‑label training** | train separate models for multiple labels from single dataset, with automatic naming based on label and language combinations, supporting both multilingual and language‑specific approaches. |
| **comprehensive metric logging** | every epoch appends to CSV logs with losses, per‑class metrics, macro F1, plus separate language‑specific performance tracking in JSON format with HuggingFace model identifiers. |
| **per‑epoch checkpoints** | lightweight checkpoint after each epoch; only the best checkpoint is retained to save disk space with configurable selection criteria. |
| **smart best‑model selection** | by default maximizes 0.7 × F1₁ + 0.3 × macro‑F1. weights and formula can be customized. |
| **automatic reinforced training** | when positive‑class F1 stays below 0.60, launches adaptive reinforced phase with class‑weighted loss, oversampling, larger batches and reduced learning rate. |
| **rescue logic for class 1** | if best normal model achieved F1₁ == 0, reinforced training considers any epoch where F1₁ > f1_1_rescue_threshold (default 0) as improvement. |
| **device auto‑detection** | native support for CUDA, Apple Silicon MPS, and CPU with automatic selection—no configuration required. |
| **data splitting with stratification** | intelligent data splitting with stratification by labels and languages, ensuring balanced representation across splits. |
| **parallel inference** | efficient parallel prediction across multiple GPUs/CPUs for high‑throughput inference. |

## 3. available models

### classic models
- **BERT variants**: bert‑base, bert‑large, and language‑specific versions (arabic, chinese, german, hindi, italian, portuguese, russian, spanish, swedish)
- **CamemBERT**: french language model
- **XLM‑RoBERTa**: cross‑lingual model

### state‑of‑the‑art models
- **DeBERTa‑v3**: xsmall (22M), base (184M), large (435M) — current SOTA for many tasks
- **RoBERTa**: base, large, distilled versions
- **ELECTRA**: small (14M), base (110M), large (335M) — excellent efficiency
- **ALBERT**: base, large, xlarge — parameter‑efficient through sharing
- **BigBird**: base, large — handles sequences up to 4096 tokens
- **Longformer**: base, large — efficient attention for long documents

### multilingual models
- **mDeBERTa‑v3**: state‑of‑the‑art multilingual model
- **XLM‑RoBERTa**: base and large versions for 100+ languages

### french‑specific models
- **CamemBERTa‑v2**: modern French RoBERTa with DeBERTa‑v3 improvements
- **CamemBERT**: base and large versions for French
- **FlauBERT**: base and large French BERT variants
- **DistilCamemBERT**: distilled version for faster inference
- **FrALBERT**: French ALBERT for parameter efficiency
- **FrELECTRA**: French ELECTRA discriminator
- **BARThez**: French sequence‑to‑sequence model

## 4. data format

### JSONL format (recommended for metadata support)

each line should be a valid JSON object:

```json
{"text": "this is a sample text", "label": 1, "id": "doc_001", "lang": "en", "source": "dataset_a"}
{"text": "ceci est un exemple", "label": 0, "id": "doc_002", "lang": "fr", "category": "politics"}
```

**required fields:**
- `text`: the input text
- `label`: the target label (string or integer)

**optional metadata fields:**
- `id`: unique identifier for the sample
- `lang`: ISO language code (enables per‑language tracking)
- any additional custom fields for analysis

### CSV format

standard CSV with headers:

```csv
text,label,id,lang,custom_field
"sample text here",1,doc_001,en,value1
"autre exemple",0,doc_002,fr,value2
```

### loading data with metadata

```python
from AugmentedSocialScientistFork.data_utils import DataLoader

# load JSONL file
samples = DataLoader.load_jsonl(
    'train.jsonl',
    text_field='text',
    label_field='label',
    id_field='id',
    lang_field='lang'
)

# load CSV file
samples = DataLoader.load_csv(
    'train.csv',
    text_column='text',
    label_column='label',
    lang_column='lang'
)

# prepare splits with language stratification
train, val, test = DataLoader.prepare_splits(
    samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by_lang=True
)
```

## 5. model selection

### automatic selection for monolingual tasks

```python
from AugmentedSocialScientistFork import ModelSelector, TaskComplexity, ResourceProfile

selector = ModelSelector()
recommendation = selector.recommend(
    task_complexity=TaskComplexity.MODERATE,  # SIMPLE, MODERATE, COMPLEX, EXTREME
    resource_profile=ResourceProfile.STANDARD, # MINIMAL, LIMITED, STANDARD, PREMIUM
    required_accuracy=0.85,
    max_inference_time=0.1  # seconds per sample
)

# instantiate recommended model
model = recommendation['model_class']()
```

### multilingual model selection

```python
from AugmentedSocialScientistFork import MultilingualModelSelector

ml_selector = MultilingualModelSelector()

# analyze language distribution and recommend
recommendation = ml_selector.recommend_model(
    texts=train_texts,  # will auto‑detect languages
    speed_priority=0.3,  # 0=accuracy, 1=speed
    min_language_coverage=0.8
)

# or specify languages directly
model_class = ml_selector.get_model_for_languages(
    languages=['en', 'fr', 'es', 'de'],
    weights=[0.4, 0.3, 0.2, 0.1]  # proportion of each language
)
```

### benchmarking models on your data

```python
# compare multiple models empirically
results = selector.benchmark_models(
    train_loader,
    val_loader,
    models_to_test=['DeBERTaV3Base', 'RoBERTaBase', 'ELECTRABase'],
    epochs=2
)
```

### dataset-level benchmarking and logging

```python
from AugmentedSocialScientistFork import BenchmarkRunner, BenchmarkConfig

runner = BenchmarkRunner(
    data_root="data/processed",
    models_root="models",
    backbone_dir="backbones",
    config=BenchmarkConfig(epochs=10, reinforced_learning=True)
)

runner.ensure_backbones_cached()  # optional; quietly skipped if offline
summaries = runner.run(multiple_datasets=True)

for run in summaries:
    print(run.dataset, run.category, run.language, run.metrics.get('macro_f1'))
```

The runner mirrors the vitrine pipeline script: it scans `training_data*` folders, routes
languages to the right backbone, logs per-language validation metrics, updates
`models_summary.csv`, and keeps structured results in memory for downstream analysis.

## 6. multi‑label training with separate models

### training separate models for multiple labels

the package supports training separate binary classifiers from multi‑label data, with intelligent naming based on labels and languages.

#### data format for multi‑label

```json
{"text": "this product is great", "sentiment": "positive", "quality": 1, "recommend": true, "lang": "en"}
{"text": "produit excellent", "sentiment": "positive", "quality": 1, "recommend": true, "lang": "fr"}
```

#### automatic multi‑label training

```python
from AugmentedSocialScientistFork.multi_label_trainer import train_multi_label_models

# train separate model for each label
models = train_multi_label_models(
    'multi_label_data.jsonl',
    label_fields=['sentiment', 'quality', 'recommend'],  # or None to auto‑detect
    train_by_language=False,  # single model per label
    multilingual=True,  # use multilingual model
    n_epochs=5,
    auto_select=True  # auto‑select best model architecture
)
# creates models: "sentiment", "quality", "recommend"
```

#### language‑specific models

```python
# train separate models per label AND language
models = train_multi_label_models(
    'multi_label_data.jsonl',
    train_by_language=True,  # separate models per language
    multilingual=False,
    n_epochs=5
)
# creates models: "sentiment_en", "sentiment_fr", "quality_en", "quality_fr", etc.
```

#### advanced multi‑label configuration

```python
from AugmentedSocialScientistFork.multi_label_trainer import (
    MultiLabelTrainer, TrainingConfig
)

# configure training
config = TrainingConfig(
    train_by_language=True,  # separate models per language
    multilingual_model=False,  # don't use multilingual models
    auto_select_model=True,  # auto‑select architecture
    n_epochs=10,
    batch_size=16,
    reinforced_learning=True,
    parallel_training=True,  # train models in parallel
    max_workers=4,
    output_dir="./specialized_models"
)

trainer = MultiLabelTrainer(config)

# load multi‑label data
samples = trainer.load_multi_label_data(
    'data.jsonl',
    label_fields=['sentiment', 'category', 'priority']
)

# train all models
trained_models = trainer.train_all_models(samples, train_ratio=0.8)

# model naming convention:
# - single model per label: "sentiment", "category", "priority"
# - models per language: "sentiment_en", "sentiment_fr", "category_en", etc.
# - output structure:
#   ./specialized_models/
#     sentiment_en/
#       - model files
#       - logs/
#     sentiment_fr/
#     category_en/
#     training_summary.json
```

#### prediction with multi‑label models

```python
import pandas as pd

# load trained models
trainer.load_trained_models('./specialized_models/training_summary.json')

# predict all labels for new texts
texts = ["new text to classify", "another example"]
languages = ["en", "en"]  # optional

predictions = trainer.predict_all_labels(
    texts=texts,
    languages=languages
)
# returns DataFrame with columns: text, sentiment_pred, category_pred, priority_pred
```

#### model naming rules
- **single multilingual model per label**: `{label_name}` (e.g., "sentiment")
- **language‑specific models**: `{label_name}_{language}` (e.g., "sentiment_en", "sentiment_fr")
- **automatic organization**: models are organized in subdirectories with their logs
- **performance tracking**: separate metrics for each label and language combination

## 7. training with metadata and language tracking

### enhanced training with per‑language metrics

```python
from AugmentedSocialScientistFork import BertBaseEnhanced
from AugmentedSocialScientistFork.data_utils import DataLoader

# load data with metadata
train_samples = DataLoader.load_jsonl('train.jsonl')
val_samples = DataLoader.load_jsonl('validation.jsonl')

# initialize model (works with any SOTA model)
model = BertBaseEnhanced(model_name='microsoft/deberta-v3-base')

# encode with metadata preservation
train_loader = model.encode_with_metadata(train_samples)
val_loader = model.encode_with_metadata(val_samples)

# train with language tracking
scores = model.run_training_enhanced(
    train_loader,
    val_loader,
    n_epochs=5,
    save_model_as="multilingual_model",
    track_languages=True  # enables per‑language metrics
)
```

**output files:**
- `training_logs/training_metrics_enhanced.csv`: overall metrics per epoch
- `training_logs/language_performance.json`: detailed per‑language metrics
- `models/multilingual_model/`: best checkpoint

### standard training (backward compatible)

```python
from AugmentedSocialScientistFork import DeBERTaV3Base

model = DeBERTaV3Base()
train_loader = model.encode(train_texts, train_labels)
val_loader = model.encode(val_texts, val_labels)

scores = model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    save_model_as="my_model",
    reinforced_learning=True
)
```

## 8. metric tracking

### standard metrics
- CSV logs in `training_logs/training_metrics.csv`
- per‑epoch: train loss, validation loss, per‑class precision/recall/F1, macro F1
- best model summary in `training_logs/best_models.csv`

### enhanced metrics with metadata
- language‑specific performance in `training_logs/language_performance.json`
- stratified analysis by any metadata field
- confusion matrices per language
- distribution statistics

### accessing performance data

```python
from AugmentedSocialScientistFork.data_utils import PerformanceTracker

tracker = PerformanceTracker()
tracker.add_batch(predictions, labels, metadata)
metrics = tracker.calculate_metrics()

# detailed report
tracker.save_detailed_report('performance_report.json')
tracker.print_summary()
```

## 9. reinforced training

triggered automatically when F1(class 1) < 0.60 after main training:
- oversampling of minority class through WeightedRandomSampler
- batch size doubled to 64, learning rate reduced to 5e‑6
- weighted cross‑entropy with pos_weight = 2.0
- full logging to `reinforced_training_metrics.csv`
- optional rescue logic for severely imbalanced cases

## 10. ensemble models

### creating multilingual ensembles

```python
from AugmentedSocialScientistFork import (
    MDeBERTaV3Base, XLMRobertaLarge,
    create_multilingual_ensemble
)

# initialize models
model1 = MDeBERTaV3Base()
model2 = XLMRobertaLarge()

# create ensemble
ensemble = create_multilingual_ensemble(
    models=[model1, model2],
    weights=[0.6, 0.4],
    voting='soft'  # or 'hard'
)

# use ensemble for prediction
predictions = ensemble.predict(test_loader)
```

## 11. data splitting with stratification

```python
from AugmentedSocialScientistFork import DataSplitter, SplitConfig

# configure splitting
config = SplitConfig(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by_label=True,
    stratify_by_language=True,
    min_samples_per_stratum=5
)

splitter = DataSplitter(config)

# create stratified splits
train, val, test = splitter.split(samples)

# or use convenience function
from AugmentedSocialScientistFork import create_stratified_splits

train, val, test = create_stratified_splits(
    samples,
    train_ratio=0.8,
    stratify_by=['label', 'lang']
)
```

## 12. parallel inference

```python
from AugmentedSocialScientistFork import parallel_predict

# run inference on multiple GPUs
predictions = parallel_predict(
    model_path="./models/best_model",
    data_loader=test_loader,
    num_gpus=2,
    batch_size=64
)

# or with multiple models
predictions = parallel_predict(
    texts=test_texts,
    model_names=['DeBERTaV3Base', 'RoBERTaBase'],
    aggregate='voting'  # or 'averaging'
)
```

## 13. configuration reference

| argument | default | purpose |
|----------|---------|---------|
| `n_epochs` | 3 | epochs in main training loop |
| `lr` | 5e‑5 | learning rate |
| `batch_size` | 32 | batch size for data loaders |
| `f1_class_1_weight` | 0.7 | weight of F1₁ in combined metric |
| `metrics_output_dir` | "./training_logs" | directory for CSV/JSON logs |
| `pos_weight` | None | class weights for loss function |
| `reinforced_learning` | False | enable adaptive reinforced phase |
| `n_epochs_reinforced` | 2 | epochs in reinforced phase |
| `rescue_low_class1_f1` | False | activate rescue logic for stalled F1₁ |
| `f1_1_rescue_threshold` | 0.0 | minimum F1₁ improvement for rescue |
| `track_languages` | True | enable per‑language performance tracking |

## 14. installation

```bash
pip install git+https://github.com/antoinelemor/AugmentedSocialScientistFork
```

### from source

```bash
git clone https://github.com/antoinelemor/AugmentedSocialScientistFork.git
cd AugmentedSocialScientistFork
pip install -r requirements.txt
pip install -e .
```

### requirements
- python 3.8+
- torch >= 2.0.0
- transformers >= 4.35.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit‑learn >= 1.0.0
- langdetect >= 1.0.9 (for multilingual features)
- tqdm >= 4.65.0
- colorama >= 0.4.6
- tabulate >= 0.9.0
- scipy >= 1.9.0
- jsonlines >= 3.1.0
- accelerate >= 0.20.0 (optional, for better performance)

### optional dependencies

```bash
# for development
pip install -e ".[dev]"

# for benchmarking visualizations
pip install -e ".[benchmarking]"
```

## 15. command‑line interface (CLI)

### interactive training CLI

the package includes a comprehensive CLI for training orchestration with multiple usage methods:

```bash
# method 1: launch interactive mode directly (no arguments needed)
python scripts/train_models.py
# → will prompt for directories and launch interactive menus

# method 2: specify directories via command line
python scripts/train_models.py --data-dir ./data --models-dir ./models --logs-dir ./logs

# method 3: direct execution with specific mode
python scripts/train_models.py --data-dir ./data --models-dir ./models --logs-dir ./logs --mode benchmark

# method 4: after package installation
pip install -e .
augmented-train --data-dir ./data --models-dir ./models --logs-dir ./logs

# method 5: show help and all options
python scripts/train_models.py --help
```

### quick start without installation

```bash
# clone the repository
git clone https://github.com/antoinelemor/AugmentedSocialScientistFork.git
cd AugmentedSocialScientistFork

# install dependencies
pip install -r requirements.txt

# launch interactive CLI (simplest method)
python scripts/train_models.py
```

when launched without arguments, the CLI will:
1. prompt for data, models, and logs directories (with defaults)
2. create directories if they don't exist
3. show interactive menu to select training mode
4. guide through configuration with prompts
5. automatically select appropriate models based on your data

### CLI modes

the CLI supports three main modes:

1. **multi‑label training**: trains separate models for each label with automatic language detection
2. **benchmark mode**: compares all appropriate models to find the best
3. **legacy mode**: backward compatibility with folder‑based data structures

### command‑line arguments

```bash
# required arguments
--data-dir DIR      # directory containing training data files
--models-dir DIR    # directory to save trained models
--logs-dir DIR      # directory to save logs and metrics

# optional arguments
--mode MODE         # training mode: interactive (default), multi-label, benchmark
--config FILE       # JSON configuration file for non-interactive mode
--quiet            # reduce output verbosity

# examples
python scripts/train_models.py \
    --data-dir ./my_data \
    --models-dir ./trained_models \
    --logs-dir ./training_logs \
    --mode benchmark
```

### programmatic usage

```python
from AugmentedSocialScientistFork import TrainingCLI

# create CLI instance
cli = TrainingCLI(
    data_dir=Path("./data"),
    models_dir=Path("./models"),
    logs_dir=Path("./logs"),
    verbose=True
)

# run interactive mode
exit_code = cli.run()
```

### non‑interactive configuration

create a JSON configuration file:

```json
{
  "mode": "multi-label",
  "data_file": "data/training_data.jsonl",
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 5e-5,
  "auto_split": true,
  "split_ratio": 0.8,
  "stratified": true,
  "reinforced_learning": true,
  "auto_select_model": true
}
```

use with:

```bash
augmented-train --data-dir ./data --models-dir ./models --logs-dir ./logs --config config.json
```

### language‑aware model selection in CLI

the CLI automatically:
- detects languages in your data
- selects appropriate models for each language
- tests models only on their supported languages
- provides language‑specific metrics

supported language‑specific models:
- **French**: CamemBERTa‑v2, CamemBERT, FlauBERT, DistilCamemBERT, FrALBERT, FrELECTRA
- **English**: RoBERTa, DeBERTa‑v3, ELECTRA, ALBERT, DistilRoBERTa
- **Multilingual**: mDeBERTa‑v3, XLM‑RoBERTa

### benchmark mode with comprehensive logging

```python
# programmatically run benchmark
from AugmentedSocialScientistFork import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    epochs=3,
    save_benchmark_csv=True,
    track_languages=True
)

runner = BenchmarkRunner(
    data_root=Path("./data"),
    models_root=Path("./models"),
    config=config
)

# test all appropriate models
best_model = runner.run_comprehensive_benchmark(
    data_path=Path("data/test.jsonl"),
    test_all_models=True,  # tests all models appropriate for detected languages
    save_detailed_log=True,  # saves benchmark_detailed_*.csv
    save_best_models_log=True  # saves benchmark_best_models_*.csv
)
```

output files include:
- `benchmark_detailed_*.csv`: all models with HuggingFace names and metrics
- `benchmark_best_models_*.csv`: summary of best models
- `benchmark_results_*.json`: complete results in JSON format

## 16. quick start examples

### simple binary classification

```python
from AugmentedSocialScientistFork import auto_select_model

# automatic model selection
ModelClass = auto_select_model(
    train_texts=texts[:100],  # sample for analysis
    resource_constraint='standard'
)

# train
model = ModelClass()
train_loader = model.encode(train_texts, train_labels)
val_loader = model.encode(val_texts, val_labels)

model.run_training(
    train_loader, val_loader,
    n_epochs=5,
    save_model_as="binary_classifier"
)
```

### multilingual classification with metadata

```python
from AugmentedSocialScientistFork import MultilingualModelSelector, BertBaseEnhanced
from AugmentedSocialScientistFork.data_utils import DataLoader

# load multilingual data
samples = DataLoader.load_jsonl('multilingual_data.jsonl')

# select best multilingual model
selector = MultilingualModelSelector()
rec = selector.recommend_model(texts=[s.text for s in samples[:1000]])

# train with language tracking
model = BertBaseEnhanced(model_name=rec.model_name)
train_loader = model.encode_with_metadata(samples)

model.run_training_enhanced(
    train_loader, val_loader,
    track_languages=True,
    save_model_as="multilingual_classifier"
)
```

### long document classification

```python
from AugmentedSocialScientistFork import LongformerBase

# for documents up to 4096 tokens
model = LongformerBase()
train_loader = model.encode(long_documents, labels)

model.run_training(
    train_loader, val_loader,
    n_epochs=3,
    save_model_as="document_classifier"
)
```

## 16. advanced usage

### custom model configuration

```python
from AugmentedSocialScientistFork import DeBERTaV3Large

model = DeBERTaV3Large()

# custom training parameters
model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    lr=2e-5,
    pos_weight=torch.tensor([1.0, 3.0]),  # weight class 1 more
    f1_class_1_weight=0.8,  # prioritize minority class
    reinforced_learning=True,
    rescue_low_class1_f1=True
)
```

### export metrics for analysis

```python
import pandas as pd
import json

# load training metrics
df = pd.read_csv('training_logs/training_metrics_enhanced.csv')

# load language performance
with open('training_logs/language_performance.json') as f:
    lang_perf = json.load(f)

# analyze performance trends
for epoch_data in lang_perf:
    epoch = epoch_data['epoch']
    for lang, metrics in epoch_data['metrics']['per_language'].items():
        print(f"epoch {epoch}, {lang}: F1={metrics['macro_f1']:.3f}")
```

## 17. model comparison guide

| model | parameters | speed | accuracy | best for |
|-------|------------|-------|----------|----------|
| ELECTRA‑small | 14M | ★★★★★ | ★★★ | edge deployment, real‑time |
| DeBERTa‑v3‑xsmall | 22M | ★★★★★ | ★★★★ | mobile, resource‑constrained |
| DistilRoBERTa | 82M | ★★★★ | ★★★★ | production systems |
| RoBERTa‑base | 125M | ★★★ | ★★★★ | general classification |
| DeBERTa‑v3‑base | 184M | ★★★ | ★★★★★ | complex tasks, nuanced understanding |
| RoBERTa‑large | 355M | ★★ | ★★★★★ | research, competitions |
| DeBERTa‑v3‑large | 435M | ★ | ★★★★★ | maximum accuracy |
| Longformer‑base | 148M | ★★ | ★★★★ | long documents (4096 tokens) |
| mDeBERTa‑v3 | 278M | ★★ | ★★★★★ | multilingual tasks |

## 18. license & citation

this project is under MIT license.

if used in academic work, please cite:
- original repository: rubingshen/AugmentedSocialScientist
- this enhanced version: antoinelemor/AugmentedSocialScientistFork

## 19. contributing

contributions welcome. please ensure:
- code follows existing style conventions
- new models include appropriate documentation
- tests pass for core functionality
- performance benchmarks are provided for new models
