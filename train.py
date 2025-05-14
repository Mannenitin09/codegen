import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from model import CodeT5Model
from dataset import create_dataloaders, get_tokenizer
from utils import (
    setup_logging,
    compute_bleu,
    compute_exact_match,
    plot_training_metrics,
    save_model_checkpoint,
    set_seed,
    count_parameters
)
from config import (
    MODEL_CONFIG,
    TRAIN_CONFIG,
    OUTPUT_DIR
)

logger = setup_logging()

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch + 1}')
    for step, batch in enumerate(progress_bar):
        logger.debug(f"Epoch {epoch + 1}, Step {step}: Moving batch to device {device}")
        batch = {k: v.to(device) for k, v in batch.items()}
        logger.debug(f"Epoch {epoch + 1}, Step {step}: Batch moved. Performing forward pass.")
        
        outputs = model(**batch)
        loss = outputs.loss
        logger.debug(f"Epoch {epoch + 1}, Step {step}: Forward pass complete. Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    avg_loss = total_loss / num_batches
    train_metrics = {} 
    return avg_loss, train_metrics

def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate the model and compute loss, BLEU, and Exact Match."""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    all_predictions = []
    all_references = [] 
    all_decoded_labels = []
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc='Validation')):
            logger.debug(f"Validation Step {step}: Moving batch to device {device}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logger.debug(f"Validation Step {step}: Loss calculated: {loss.item():.4f}")

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MODEL_CONFIG['max_target_length'],
                num_beams=MODEL_CONFIG.get('num_beams', 4),
                early_stopping=True
            )
            logger.debug(f"Validation Step {step}: Predictions generated.")
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels[labels == -100] = tokenizer.pad_token_id 
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            logger.debug(f"Validation Step {step}: Predictions and labels decoded.")

            all_predictions.extend(decoded_preds)
            all_references.extend([[label] for label in decoded_labels]) 
            all_decoded_labels.extend(decoded_labels) 
            
    avg_loss = total_loss / num_batches
    logger.info(f"Average validation loss: {avg_loss:.4f}")
    
    metrics = {'bleu': 0.0, 'exact_match': 0.0} 
    try:
        logger.info("Computing validation BLEU score...")
        metrics['bleu'] = compute_bleu(all_predictions, all_references)
        logger.info("Computing validation Exact Match score...")
        metrics['exact_match'] = compute_exact_match(all_predictions, all_decoded_labels)
        logger.info("Validation metrics computed.")
    except Exception as e:
        logger.error(f"Error computing validation metrics: {e}", exc_info=True)
        
    return avg_loss, metrics

def train():
    """Main training loop."""
    logger.info("--- Starting Training --- ")
    
    set_seed(TRAIN_CONFIG.get("seed", 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    logger.info("Initializing model...")
    model = CodeT5Model.from_pretrained(MODEL_CONFIG["model_name"])
    model.to(device)
    logger.info(f"Model loaded: {MODEL_CONFIG['model_name']}")
    logger.info(f"Total trainable parameters: {count_parameters(model):,}")

    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader, _ = create_dataloaders(tokenizer)
    logger.info(f"Dataloaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
    
    logger.info("Initializing optimizer...")
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"]["initial"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
    except Exception as e:
        logger.error(f"Error initializing optimizer: {e}", exc_info=True)
        raise
    
    logger.info("Initializing learning rate scheduler...")
    try:
        num_training_steps = len(train_dataloader) * TRAIN_CONFIG["num_train_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
            num_training_steps=num_training_steps
        )
    except Exception as e:
        logger.error(f"Error initializing scheduler: {e}", exc_info=True)
        raise

    train_losses = []
    val_losses = []
    val_bleu_history = []
    val_exact_match_history = [] 
    lr_history = [] 
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("--- Starting Training Loop ---")
    for epoch in range(TRAIN_CONFIG["num_train_epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{TRAIN_CONFIG['num_train_epochs']}")
        
        train_loss, _ = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            device, epoch
        )
        
        val_loss, val_metrics = validate(model, val_dataloader, tokenizer, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bleu_history.append(val_metrics.get('bleu', 0.0))
        val_exact_match_history.append(val_metrics.get('exact_match', 0.0))
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        
        logger.info(f"--- Epoch {epoch + 1} Summary ---")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val BLEU: {val_metrics.get('bleu', 0.0):.4f}") 
        logger.info(f"  Val Exact Match: {val_metrics.get('exact_match', 0.0):.4f}")
        logger.info(f"  Current LR: {current_lr:.6f}")
        logger.info("-----------------------")
        
        if val_loss < best_val_loss:
            logger.info(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
            best_val_loss = val_loss
            patience_counter = 0
            save_model_checkpoint(
                model, optimizer, epoch, val_loss,
                val_metrics, OUTPUT_DIR, is_best=True
            )
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{TRAIN_CONFIG['early_stopping_patience']}")
        
        try:
            plot_training_metrics(
                train_losses, val_losses,
                {}, 
                {'bleu': val_bleu_history, 'exact_match': val_exact_match_history},
                lr_history, 
                OUTPUT_DIR
            )
        except Exception as e:
             logger.error(f"Failed to plot metrics for epoch {epoch+1}: {e}", exc_info=True)

        if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            logger.info(
                f"Early stopping triggered after {patience_counter} epochs "
                f"without improvement on validation loss."
            )
            break
        
    logger.info("--- Training Finished ---")

if __name__ == "__main__":
    train()
