# Fine-Tuning Implementation Documentation

## Overview

**Notebook**: `fine_tune_improved.ipynb`

**Purpose**: Fine-tune a single transformer model for metaphor detection on Catalan text using token classification. The notebook evaluates the base model before fine-tuning and compares performance metrics to demonstrate the impact of fine-tuning.

**Task Type**: Token Classification (Named Entity Recognition-style labeling)

**Dataset**: `mariadelcarmenramirez/metaphor-catalan-iter1` (Hugging Face Datasets)

---

## Architecture & Workflow

### 1. **Configuration Phase**
- Single model selection (configurable checkpoint and name)
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- Reproducibility settings (random seed)
- Output paths and Hugging Face Hub integration

### 2. **Data Loading & Validation**
- Load pre-split dataset (train/validation/test) from Hugging Face Hub
- Extract label information (label list, id2label, label2id mappings)
- Validate label column presence
- Display dataset statistics and sample examples

### 3. **Class Imbalance Analysis**
- Compute token-level label distribution across training set
- Identify potential class imbalance issues
- Critical for interpreting model performance (metaphors are typically rare)

### 4. **Data Processing**
- **Tokenization**: Subword tokenization using model-specific tokenizer
- **Label Alignment**: Map word-level labels to subword tokens
  - First subword of each word receives the label
  - Subsequent subwords receive `-100` (ignored in loss calculation)
  - Special tokens ([CLS], [SEP], [PAD]) also receive `-100`

### 5. **Base Model Evaluation**
- Load pretrained model with randomly initialized classification head
- Evaluate on test set **before any training**
- Establish baseline metrics (precision, recall, F1, accuracy)
- Per-label metrics computed for detailed analysis

### 6. **Fine-Tuning**
- Reload fresh model from checkpoint
- Train on training set, validate on validation set
- Early stopping: best model selected based on F1 score on validation set
- Checkpointing: saves top 2 models during training
- Mixed precision (FP16) enabled automatically if GPU available

### 7. **Fine-Tuned Model Evaluation**
- Evaluate best checkpoint on test set
- Compute detailed metrics with per-label breakdown
- Calculate improvement over base model

### 8. **Results Export**
- Save all metrics to CSV file (single-row format)
- Include training configuration, seed, and status
- Error handling: captures failures and logs error messages

### 9. **Practical Testing**
- Load best model and create inference pipeline
- Test on sample Catalan sentences
- Display detected metaphors with confidence scores

---

## Configuration Parameters

### Dataset Configuration
```python
DATASET_NAME = "mariadelcarmenramirez/metaphor-catalan-iter1"
```

### Model Selection
```python
MODEL_CHECKPOINT = "projecte-aina/roberta-large-ca-v2"  # Model from Hugging Face
MODEL_NAME = "roberta"  # Short name for output directories
```

### Training Hyperparameters
```python
TRAINING_CONFIG = {
    "learning_rate": 2e-5,                    # Adam learning rate
    "per_device_train_batch_size": 8,        # Batch size for training
    "per_device_eval_batch_size": 8,         # Batch size for evaluation
    "num_train_epochs": 5,                   # Number of training epochs
    "weight_decay": 0.01,                    # L2 regularization
    "gradient_accumulation_steps": 1,        # Accumulate gradients (increase if OOM)
}
```

### Reproducibility
```python
SEED = 42  # Fixed random seed for reproducibility
```

### Hugging Face Hub Integration
```python
PUSH_TO_HUB = True
HUB_USERNAME = "mariadelcarmenramirez"
```

### Output Paths
```python
OUTPUT_DIR = f"./{MODEL_NAME}-metaphor-detection-cat"
RESULTS_CSV = f"model_comparison_results_{MODEL_NAME}.csv"
```

---

## Key Functions

### `tokenize_and_align_labels(examples, tokenizer)`
**Purpose**: Convert word-level tokens and labels to subword-level tokens and labels.

**Process**:
1. Tokenize word-level tokens into subwords
2. Map word indices to subword positions using `word_ids()`
3. Assign labels:
   - `-100` for special tokens (ignored in loss)
   - Original label for first subword of each word
   - `-100` for subsequent subwords of same word
4. Validate: raises error if token count ≠ label count

**Why Needed**: Transformers use subword tokenization (BPE), which splits words into multiple tokens. Labels must align correctly.

### `compute_metrics(eval_pred)`
**Purpose**: Calculate overall evaluation metrics during training.

**Returns**:
- `precision`: Overall precision across all labels
- `recall`: Overall recall across all labels
- `f1`: Overall F1 score (harmonic mean of precision/recall)
- `accuracy`: Overall token-level accuracy

**Library**: Uses `seqeval` metric (standard for token classification)

### `compute_detailed_metrics(eval_pred)`
**Purpose**: Calculate metrics with per-label breakdown for detailed analysis.

**Returns**:
- Overall metrics (precision, recall, F1, accuracy)
- Per-label metrics (precision, recall, F1 for each label type)

**Use Case**: Identify which labels are well-learned vs poorly-learned

### `evaluate_model(model, tokenizer, tokenized_test_dataset, model_name)`
**Purpose**: Evaluate a model on test set using Trainer API.

**Process**:
1. Create temporary TrainingArguments (evaluation-only)
2. Initialize Trainer with model and data collator
3. Run evaluation on test dataset
4. Return detailed metrics dictionary

**Note**: Creates temporary output directory for Trainer requirements

---

## Training Process

### Phase 1: Base Model Evaluation
1. Load pretrained model with classification head (random initialization)
2. Tokenize entire dataset
3. Evaluate on test set
4. Record baseline metrics

### Phase 2: Fine-Tuning
1. **Setup**: Fresh model instance, same pretrained weights
2. **Training Arguments**:
   - Evaluate every epoch
   - Save checkpoint every epoch (keep best 2)
   - Load best model at end (based on F1 score)
   - Log every 100 steps
   - FP16 mixed precision (if GPU available)
   - Fixed seeds for reproducibility

3. **Training Loop**: Handled by Hugging Face Trainer
   - Automatic batching, gradient accumulation
   - Validation after each epoch
   - Best checkpoint selection based on validation F1

4. **Saving**: Model, tokenizer, and trainer state saved to `OUTPUT_DIR`

### Phase 3: Final Evaluation
1. Load best checkpoint (already loaded by Trainer)
2. Run inference on test set
3. Compute detailed metrics with per-label breakdown
4. Calculate improvement: `fine-tuned F1 - base F1`

---

## Evaluation Metrics

### Overall Metrics
- **Precision**: TP / (TP + FP) - How many predicted metaphors are correct?
- **Recall**: TP / (TP + FN) - How many actual metaphors are detected?
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: Correct predictions / Total predictions (token-level)

### Per-Label Metrics
- Individual precision, recall, F1 for each label type (e.g., "O", "METAPHOR")
- Critical for understanding class-specific performance
- Exposes issues with imbalanced classes

### Why F1 is Primary Metric
- **Class Imbalance**: Non-metaphors ("O") typically dominate (~95%+)
- **Accuracy Misleading**: Model predicting all "O" achieves high accuracy but fails the task
- **F1 Balance**: Equally weights precision and recall, penalizes models that ignore minority class

---

## Output Files

### 1. Model Checkpoint Directory
**Path**: `./{MODEL_NAME}-metaphor-detection-cat/`

**Contents**:
- `config.json`: Model configuration
- `model.safetensors` or `pytorch_model.bin`: Model weights
- `tokenizer_config.json`, `vocab.json`, `merges.txt`: Tokenizer files
- `training_args.bin`: Training configuration
- `trainer_state.json`: Training history
- Checkpoint subdirectories (if intermediate checkpoints saved)

### 2. Results CSV
**Path**: `model_comparison_results_{MODEL_NAME}.csv`

**Columns**:
- `status`: "success" or "failed"
- `model`: Model short name
- `checkpoint`: Full model checkpoint path
- `seed`: Random seed used
- `gradient_accumulation_steps`: Gradient accumulation setting
- `base_precision`, `base_recall`, `base_f1`, `base_accuracy`: Base model metrics
- `finetuned_precision`, `finetuned_recall`, `finetuned_f1`, `finetuned_accuracy`: Fine-tuned model metrics
- `improvement_f1`: F1 score improvement (fine-tuned - base)
- `error`: Error message (if status = "failed")

### 3. Temporary Directories
**Path**: `./tmp_eval_{model_name}/`

**Purpose**: Required by Trainer API for evaluation-only runs

**Note**: Can be safely deleted after execution; consider using `tempfile.TemporaryDirectory()` for auto-cleanup

---

## Error Handling

### Try-Except Wrapper
The entire training pipeline is wrapped in a try-except block:

```python
try:
    # Load, train, evaluate
    results_row.update({"status": "success", ...})
except Exception as e:
    # Log error
    results_row["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()
```

**Benefits**:
- Training failures don't crash the notebook
- Errors are captured in results CSV
- Full traceback printed for debugging

### Common Errors
1. **Out of Memory (OOM)**: Increase `gradient_accumulation_steps`, reduce batch size
2. **Label Mismatch**: Validation error catches data inconsistencies
3. **Hub Authentication**: Ensure `huggingface-cli login` completed before `PUSH_TO_HUB=True`

---

## Reproducibility Features

### Random Seed Setting
```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)  # Transformers library seed
```

### Seed in Training Arguments
```python
TrainingArguments(
    seed=SEED,
    data_seed=SEED,
    ...
)
```

**Impact**: Ensures identical results across runs (same data shuffling, weight initialization, dropout)

**Limitation**: GPU operations may introduce minor variations; use deterministic algorithms for exact reproducibility (slower)

---

## Label Format

### Expected Format
```python
{
    "tokens": ["Van", "deixar", "de", "visitar", ...],
    "tags": [0, 1, 0, 0, ...]
}
```

- `tokens`: List of word-level tokens (already tokenized)
- `tags`: List of label IDs, one per token
- Label mapping: `{0: "O", 1: "METAPHOR"}` (example)

### BIO Format (Alternative)
Some datasets use Begin-Inside-Outside tagging:
- `B-METAPHOR`: Beginning of metaphor span
- `I-METAPHOR`: Inside metaphor span
- `O`: Outside (non-metaphor)

The notebook auto-detects label names from dataset features.

---

## Inference Pipeline

### Loading Fine-Tuned Model
```python
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)

metaphor_detector = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)
```

### Usage
```python
results = metaphor_detector("El temps vola quan t'ho passes bé.")
# Returns: List of detected entities with word, label, score, start, end
```

### Aggregation Strategy
- `"simple"`: Merge consecutive tokens with same label
- Groups subword tokens back into words
- Returns word-level predictions instead of subword-level

---

## Hardware Requirements

### Minimum
- **CPU**: Any modern CPU
- **RAM**: 16 GB (for large models like RoBERTa-large)
- **Training Time**: Hours per epoch (depends on dataset size)

### Recommended
- **GPU**: NVIDIA GPU with 8+ GB VRAM (enables FP16 mixed precision)
- **RAM**: 32 GB
- **Training Time**: Minutes per epoch with GPU

### Memory Optimization
- **Reduce batch size**: Lower `per_device_train_batch_size`
- **Gradient accumulation**: Increase `gradient_accumulation_steps`
- **Smaller model**: Use base models instead of large models

---

## Extension Opportunities

### Multiple Models
- Modify configuration cell to loop over multiple models
- Compare performance across different architectures
- Original notebook version supports this (see `MODELS_TO_TRAIN` list)

### Hyperparameter Tuning
- Implement grid search or random search over learning rates, batch sizes
- Use Optuna or Ray Tune for automated hyperparameter optimization
- Track experiments with Weights & Biases or TensorBoard

### Early Stopping
```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

### Class Weights
Address class imbalance by weighting loss:
```python
class_weights = compute_class_weight('balanced', classes=[0, 1], y=all_labels)
# Implement custom Trainer or loss function with weights
```

### Data Augmentation
- Back-translation for generating additional training examples
- Synonym replacement for metaphorical expressions
- Contextual word embeddings perturbation

---

## Best Practices

### Before Training
1. ✅ Check label distribution (class imbalance)
2. ✅ Validate data format (tokens match labels)
3. ✅ Set reproducibility seeds
4. ✅ Monitor GPU memory usage

### During Training
1. ✅ Monitor validation metrics each epoch
2. ✅ Watch for overfitting (train F1 >> validation F1)
3. ✅ Check training logs for anomalies

### After Training
1. ✅ Evaluate on held-out test set
2. ✅ Analyze per-label metrics (which labels fail?)
3. ✅ Test on real examples (sanity check)
4. ✅ Save results and model checkpoints

### Documentation
1. ✅ Record hyperparameters used
2. ✅ Save training logs and metrics
3. ✅ Document model version and dataset version
4. ✅ Include random seed in results

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**:
- Reduce `per_device_train_batch_size` to 4 or 2
- Increase `gradient_accumulation_steps` to 2 or 4
- Use smaller model (e.g., base instead of large)

### Issue: Poor Performance (Low F1)
**Diagnosis**:
- Check class imbalance (>90% one class?)
- Review per-label metrics (minority class F1 = 0?)
- Verify data quality (correct labels?)

**Solution**:
- Increase training epochs (5 → 10)
- Adjust learning rate (2e-5 → 3e-5)
- Implement class weighting

### Issue: Model Predicts All One Class
**Diagnosis**: Severe class imbalance, model learned trivial solution

**Solution**:
- Implement focal loss or class weights
- Oversample minority class
- Use different threshold for classification

### Issue: No Improvement After Fine-Tuning
**Diagnosis**: 
- Model already saturated (base model very strong)
- Learning rate too high/low
- Insufficient training data

**Solution**:
- Try different architecture
- Increase training epochs
- Collect more annotated data
- Use domain-specific pretrained model

---

## References

### Libraries
- **Transformers**: Hugging Face library for transformer models
- **Datasets**: Hugging Face library for dataset management
- **Evaluate**: Hugging Face library for evaluation metrics
- **seqeval**: Evaluation library for sequence labeling tasks

### Documentation
- [Hugging Face Token Classification Guide](https://huggingface.co/docs/transformers/tasks/token_classification)
- [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

### Models Used (Example)
- **RoBERTa-large-ca-v2**: Catalan RoBERTa model by Projecte AINA
- **mDeBERTa-v3-base**: Multilingual DeBERTa by Microsoft
- **mdeberta-base-metaphor-detection-es**: Spanish metaphor detection model by HiTZ

---

## Summary

This notebook provides a complete, production-ready pipeline for fine-tuning transformer models on token classification tasks with comprehensive evaluation and error handling. Key features include:

- ✅ Before/after comparison to demonstrate training impact
- ✅ Reproducible results with fixed random seeds
- ✅ Detailed metrics with per-label breakdown
- ✅ Class imbalance analysis
- ✅ Error handling and logging
- ✅ Practical inference examples
- ✅ Hugging Face Hub integration
- ✅ CSV results export for experiment tracking

The implementation follows best practices for NLP model training and serves as a template for similar token classification tasks.
