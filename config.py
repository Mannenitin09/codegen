import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Create directories if they don't exist
for dir_path in [CACHE_DIR, OUTPUT_DIR, CHECKPOINT_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "max_source_length": 64,
    "max_target_length": 64,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": {
        "initial": 5e-5,
        "min": 1e-6,
    },
    "fp16": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 1e-4,
}

# Training configuration
TRAIN_CONFIG = {
    "num_train_epochs": 20,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "learning_rate": {
        "initial": 5e-5,
        "min": 1e-6,
    },
    "fp16": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 1e-4,
}

# Phase-wise training (if used)
PHASE_CONFIG = {
    "phase1": {
        "freeze_encoder": True,
        "freeze_decoder": True,
        "train_task_head": True,
        "epochs": 5,
    },
    "phase2": {
        "freeze_encoder": False,
        "freeze_decoder": False,
        "train_task_head": True,
        "epochs": 15,
    }
}

# Data processing
DATA_CONFIG = {
    "train_size": 0.8,
    "val_size": 0.1,
    "test_size": 0.1,
    "max_train_samples": None,
    "seed": 42,
}

# Tokenizer settings
TOKENIZER_CONFIG = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
}

# Evaluation configuration
EVAL_CONFIG = {
    "metrics": ["bleu", "codebleu", "exact_match"],
    
    # Generation parameters
    "generate_kwargs": {
        "max_length": 128,
        "num_beams": 10,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "output_attentions": True,
        "output_scores": True,
    },
    
    # CodeBLEU weights
    "codebleu_weights": {
        "ast": 0.4,
        "dataflow": 0.3,
        "controlflow": 0.3,
    },
    
    # Attention visualization
    "attention_viz": {
        "enabled": True,
        "figsize": (10, 8),
        "cmap": "viridis",
        "dpi": 100,
    },
    
    # Interactive mode
    "interactive": {
        "num_return_sequences": 3,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    
    # Syntax validation
    "syntax_validation": {
        "enabled": True,
        "show_errors": True,
        "validate_imports": False,
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
    "log_file_name_pattern": "{timestamp}.log",
    "log_every_n_steps": 100,
    "save_every_n_epochs": 1,
    "evaluation_strategy": "epoch",
}
