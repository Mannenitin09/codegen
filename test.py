import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

from dataset import create_dataloaders, get_tokenizer
from utils import setup_logging, compute_bleu, compute_exact_match, set_seed
from config import OUTPUT_DIR, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

logger = setup_logging()

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, tokenizer, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on the test set."""
    model.eval()
    predictions = []
    references = [] # For BLEU: List[List[str]]
    all_decoded_labels = [] # For Exact Match: List[str]
    
    progress_bar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Keep labels on device for reference
            
            # Generate predictions
            # Adjust generation parameters as needed (from config or defaults)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MODEL_CONFIG['max_target_length'], 
                num_beams=MODEL_CONFIG.get('num_beams', 4), # Use config or default
                early_stopping=True
            )
            
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode labels (handle padding token ID)
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            # References for BLEU need to be List[List[str]]
            references.extend([[label] for label in decoded_labels])
            # Accumulate labels for exact match
            all_decoded_labels.extend(decoded_labels)
            
    # Compute metrics
    bleu_score = compute_bleu(predictions, references)
    # Use accumulated labels (all_decoded_labels) for exact match
    exact_match_score = compute_exact_match(predictions, all_decoded_labels)
    
    metrics = {
        'bleu': bleu_score,
        'exact_match': exact_match_score
    }
    return metrics

def test():
    """Main testing function."""
    try:
        logger.info("Starting evaluation...")
        
        # Set seed
        set_seed(42)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize tokenizer
        tokenizer = get_tokenizer()
        
        # Initialize base T5 model using the original model name from config
        logger.info(f"Initializing base model: {MODEL_CONFIG['model_name']}")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])
        
        # Define checkpoint path
        checkpoint_path =  "outputs/checkpoints/best_model.pt"
        # if not checkpoint_path.exists():
        #     logger.error(f"Best model checkpoint not found at: {checkpoint_path}")
        #     return
        
        # Load checkpoint
        logger.info(f"Loading model checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle potential DataParallel wrapping if saved that way
            state_dict = checkpoint['model_state_dict']
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            # Load the state dict into the base model
            logger.info("Attempting to load state dict...")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} successfully.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            return
        
        model.to(device)
        model.eval() # Ensure model is in eval mode
        
        # Create test dataloader
        _, _, test_dataloader = create_dataloaders(tokenizer)
        logger.info(f"Created test dataloader: {len(test_dataloader)} batches")
        
        # Evaluate
        logger.info("Starting model evaluation...")
        metrics = evaluate_model(model, test_dataloader, tokenizer, device)
        logger.info("Model evaluation finished.")
        
        # Log results
        logger.info("--- Evaluation Results ---")
        logger.info(f"  BLEU Score: {metrics['bleu']:.4f}")
        logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
        logger.info("--------------------------")
        logger.info("Successfully completed evaluation.")

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)

if __name__ == "__main__":
    test()
