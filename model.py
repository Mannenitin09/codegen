import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any
from config import MODEL_CONFIG
import os

class CodeT5Model(T5ForConditionalGeneration):
    """CodeT5 model wrapper inheriting from T5ForConditionalGeneration.

    Inherits forward and generate methods directly.
    """

    def __init__(self, config=None):
        """Initialize the model configuration."""
        if config is None:
            config = T5Config.from_pretrained(MODEL_CONFIG["model_name"])
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'CodeT5Model':
        """Load from a pretrained T5 checkpoint or a fine-tuned checkpoint.

        Uses the parent class's from_pretrained method.
        """
        return super().from_pretrained(model_name_or_path, **kwargs)

    # save_pretrained is inherited from the parent class.
