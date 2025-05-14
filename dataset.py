import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
from pathlib import Path
from config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    DATA_DIR
)

class CodeGenerationDataset(Dataset):
    """Dataset for code generation task using CodeT5."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_source_length: int = MODEL_CONFIG["max_source_length"],
        max_target_length: int = MODEL_CONFIG["max_target_length"],
        split: str = "train"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.split = split
        
        self.examples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load and preprocess the dataset."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            data = [data]
            
        processed_examples = []
        for item in data:
            intent = item.get('rewritten_intent') or item.get('intent')
            snippet = item.get('snippet')
            
            if not intent or not snippet:
                continue
                
            processed_examples.append({
                'intent': intent,
                'snippet': snippet
            })
            
        if self.split == "train" and DATA_CONFIG["max_train_samples"]:
            processed_examples = processed_examples[:DATA_CONFIG["max_train_samples"]]
            
        return processed_examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        model_inputs = self.tokenizer(
            example["intent"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example["snippet"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        label_ids = labels["input_ids"]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100  # Important: we need to prepare the labels by replacing padding token id by -100
            
        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": label_ids.squeeze()
        }

def create_dataloaders(
    tokenizer: AutoTokenizer,
    train_batch_size: int = MODEL_CONFIG["train_batch_size"],
    eval_batch_size: int = MODEL_CONFIG["eval_batch_size"]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_dataset = CodeGenerationDataset(
        data_path=str(DATA_DIR / "conala" / "train.json"),
        tokenizer=tokenizer,
        split="train"
    )
    
    val_dataset = CodeGenerationDataset(
        data_path=str(DATA_DIR / "conala" / "test.json"),
        tokenizer=tokenizer,
        split="validation"
    )
    
    test_dataset = CodeGenerationDataset(
        data_path=str(DATA_DIR / "conala" / "test.json"),
        tokenizer=tokenizer,
        split="test"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def get_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    return tokenizer
