# Configuration Files

This directory contains configuration files that define model training parameters and hyperparameters for the MC Classifier Pipeline. Configuration files are JSON-formatted and specify which models to train and their respective settings.

## File Structure

- [`quick_test.json`](quick_test.json) - A lightweight configuration for testing with fast-training models
- Additional configuration files can be created for different experimental scenarios

## Configuration Format

Configuration files follow this general structure:

```json
{
  "models": [
    {
      "name": "model_identifier",
      "description": "Optional description of the model configuration",
      "model_type": "ModelClassName",
      "model_params": {
        "parameter1": "value1",
        "parameter2": "value2"
      }
    }
  ]
}
```

**Required Fields:**
- `name` (string): Unique identifier for the model configuration
- `model_type` (string): The model class to use (see supported types below)
- `model_params` (object): Parameters specific to the model type

**Optional Fields:**
- `description` (string): Human-readable description of the model configuration

## Supported Model Types

### BertFineTune

BERT-based text classification using HuggingFace Transformers.

**Required Parameters:**
- `model_name` (string): HuggingFace model identifier (e.g., "bert-base-uncased", "distilbert-base-uncased")

**Optional Parameters:**
- `num_train_epochs` (int): Number of training epochs (default: 3)
- `learning_rate` (float): Learning rate for optimizer (default: 2e-5)
- `per_device_train_batch_size` (int): Training batch size per device (default: 16)
- `per_device_eval_batch_size` (int): Evaluation batch size per device (default: 16)
- `weight_decay` (float): Weight decay for regularization (default: 0.01)
- `warmup_steps` (int): Number of warmup steps for learning rate scheduler (default: 0)
- `logging_steps` (int): Number of steps between logging outputs (default: 500)
- `eval_steps` (int): Number of steps between evaluations (default: 500)
- `save_steps` (int): Number of steps between model saves (default: 500)
- `evaluation_strategy` (string): When to run evaluation ("steps", "epoch", "no") (default: "epoch")
- `save_strategy` (string): When to save model ("steps", "epoch", "no") (default: "epoch")
- `load_best_model_at_end` (bool): Whether to load best model at end of training (default: true)
- `metric_for_best_model` (string): Metric to use for best model selection (default: "eval_loss")
- `greater_is_better` (bool): Whether higher metric values are better (default: false)
- `gradient_accumulation_steps` (int): Number of steps to accumulate gradients (default: 1)
- `max_grad_norm` (float): Maximum gradient norm for clipping (default: 1.0)
- `seed` (int): Random seed for reproducibility (default: 42)
- `fp16` (bool): Whether to use 16-bit floating point precision (default: false)
- `dataloader_num_workers` (int): Number of data loading workers (default: 0)
- `remove_unused_columns` (bool): Whether to remove unused columns from dataset (default: true)
- `label_smoothing_factor` (float): Label smoothing factor (default: 0.0)
- `logging_dir` (string): Directory for tensorboard logs (optional)
- `report_to` (array): List of reporting tools (e.g., ["tensorboard", "wandb"]) (default: none)

**Example Configuration:**
```json
{
  "name": "distilbert_classifier",
  "description": "DistilBERT with custom hyperparameters",
  "model_type": "BertFineTune",
  "model_params": {
    "model_name": "distilbert-base-uncased",
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": true,
    "seed": 42
  }
}
```

### SklearnMultinomialNaiveBayes

Scikit-learn based Multinomial Naive Bayes with TF-IDF vectorization.

**All Parameters Optional:**

**TF-IDF Vectorizer Parameters:**
- `ngram_range` (array): Range of n-values for n-grams [min_n, max_n] (default: [1, 1])
- `min_df` (int/float): Minimum document frequency for terms (default: 1)
- `max_df` (int/float): Maximum document frequency for terms (default: 1.0)
- `max_features` (int): Maximum number of features to extract (default: null - no limit)
- `stop_words` (string/array): Stop words to remove ("english", array of words, or null) (default: null)
- `lowercase` (bool): Convert text to lowercase (default: true)
- `token_pattern` (string): Regular expression for token extraction (default: "(?u)\\b\\w\\w+\\b")
- `binary` (bool): If true, all non-zero term counts set to 1 (default: false)
- `dtype` (string): Data type for the matrix ("float32", "float64") (default: "float64")
- `norm` (string): Norm used to normalize term vectors ("l1", "l2", null) (default: "l2")
- `use_idf` (bool): Enable inverse-document-frequency reweighting (default: true)
- `smooth_idf` (bool): Smooth idf weights by adding 1 to document frequencies (default: true)
- `sublinear_tf` (bool): Apply sublinear tf scaling (default: false)

**Multinomial Naive Bayes Parameters:**
- `alpha` (float): Additive (Laplace/Lidstone) smoothing parameter (default: 1.0)
- `fit_prior` (bool): Whether to learn class prior probabilities (default: true)
- `class_prior` (array): Prior probabilities of classes (default: null - uniform)

**Example Configuration:**
```json
{
  "name": "naive_bayes_advanced",
  "description": "Naive Bayes with bigrams and English stop words",
  "model_type": "SklearnMultinomialNaiveBayes",
  "model_params": {
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 10000,
    "stop_words": "english",
    "lowercase": true,
    "binary": false,
    "norm": "l2",
    "use_idf": true,
    "smooth_idf": true,
    "sublinear_tf": false,
    "alpha": 0.1,
    "fit_prior": true,
    "class_prior": null
  }
}
```

## Usage Examples

### Training Specific Models

To train only specific models from a configuration file:

```bash
python -m mc_classifier_pipeline.trainer \
    --experiment-dir experiments/project_1/20250806_103847 \
    --models-config configs/quick_test.json \
    --model-names fast_bert quick_nb
```

### Training All Models

To train all models defined in a configuration file:

```bash
python -m mc_classifier_pipeline.trainer \
    --experiment-dir experiments/project_1/20250806_103847 \
    --models-config configs/quick_test.json
```

### End-to-End Pipeline

To run the complete pipeline (preprocessing + training + evaluation):

```bash
python -m mc_classifier_pipeline.model_orchestrator \
    --project-id 1 \
    --train-ratio 0.7 \
    --output-dir experiments \
    --target-label 'Analysis' \
    --models-config configs/quick_test.json
```


## Configuration Scenarios

### 1. Quick Testing Configuration

For rapid prototyping and testing:

```json
{
  "models": [
    {
      "name": "quick_nb",
      "description": "Fast Naive Bayes baseline",
      "model_type": "SklearnMultinomialNaiveBayes",
      "model_params": {
        "ngram_range": [1, 1],
        "min_df": 1,
        "alpha": 1.0
      }
    },
    {
      "name": "fast_bert",
      "description": "Quick BERT test with minimal epochs",
      "model_type": "BertFineTune",
      "model_params": {
        "model_name": "distilbert-base-uncased",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8
      }
    }
  ]
}
```

### 2. Production Configuration

For thorough model comparison with multiple variants:

```json
{
  "models": [
    {
      "name": "nb_unigram",
      "description": "Unigram Naive Bayes with stop words",
      "model_type": "SklearnMultinomialNaiveBayes",
      "model_params": {
        "ngram_range": [1, 1],
        "min_df": 2,
        "max_df": 0.95,
        "alpha": 1.0,
        "stop_words": "english"
      }
    },
    {
      "name": "nb_bigram",
      "description": "Bigram Naive Bayes with tuned alpha",
      "model_type": "SklearnMultinomialNaiveBayes",
      "model_params": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "alpha": 0.1,
        "stop_words": "english"
      }
    },
    {
      "name": "bert_base",
      "description": "BERT base model with F1 optimization",
      "model_type": "BertFineTune",
      "model_params": {
        "model_name": "bert-base-uncased",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "evaluation_strategy": "epoch",
        "metric_for_best_model": "eval_f1",
        "greater_is_better": true
      }
    }
  ]
}
```

### 3. Hyperparameter Tuning Configuration

For systematic hyperparameter exploration:

```json
{
  "models": [
    {
      "name": "nb_alpha_001",
      "description": "Naive Bayes with alpha=0.01",
      "model_type": "SklearnMultinomialNaiveBayes",
      "model_params": {
        "alpha": 0.01
      }
    },
    {
      "name": "nb_alpha_010",
      "description": "Naive Bayes with alpha=0.1",
      "model_type": "SklearnMultinomialNaiveBayes", 
      "model_params": {
        "alpha": 0.1
      }
    },
    {
      "name": "nb_alpha_100",
      "description": "Naive Bayes with alpha=1.0",
      "model_type": "SklearnMultinomialNaiveBayes",
      "model_params": {
        "alpha": 1.0
      }
    }
  ]
}
```

## Configuration Validation

The pipeline validates configuration files and will report errors for:

- Missing required fields (`name`, `model_type`, `model_params`)
- Invalid model types (must be one of: `BertFineTune`, `SklearnMultinomialNaiveBayes`)
- Missing required parameters for specific model types (e.g., `model_name` for BertFineTune)
- Invalid parameter values or types
- Malformed JSON syntax

## Creating Custom Configurations

1. Copy [`quick_test.json`](quick_test.json) as a starting point
2. Modify the model names, types, and parameters as needed
3. Add meaningful descriptions for each model configuration
4. Validate the JSON syntax using a JSON validator
5. Test with a small experiment before running full training

## Integration with Pipeline

Configuration files are used by:

- [`trainer.py`](../src/mc_classifier_pipeline/trainer.py) - Direct model training
- [`model_orchestrator.py`](../src/mc_classifier_pipeline/model_orchestrator.py) - End-to-end pipeline
- [`evaluation.py`](../src/mc_classifier_pipeline/evaluation.py) - Evaluates models trained from configurations

The trained models will be saved with metadata including the original configuration name and parameters for reproducibility.
