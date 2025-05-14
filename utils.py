import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import LOGGING_CONFIG, OUTPUT_DIR

def setup_logging(name: str = "code_generation") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(LOGGING_CONFIG["log_level"])
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGGING_CONFIG["log_level"])
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        log_dir = OUTPUT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(LOGGING_CONFIG["log_level"])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def compute_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """Compute average sentence BLEU score using NLTK.

    Args:
        predictions: List of predicted strings.
        references: List of lists of reference strings (NLTK format: List[List[List[str]]]).

    Returns:
        Average BLEU score.
    """ 
    smoothie = SmoothingFunction().method4 
    total_bleu = 0.0
    count = 0
    
    if len(predictions) != len(references):
        logging.getLogger("code_generation").error(f"BLEU: Preds ({len(predictions)}) and refs ({len(references)}) length mismatch.")
        return 0.0
        
    for pred, ref_list in zip(predictions, references):
        try:
            tokenized_pred = str(pred).split()
            tokenized_refs = [str(r).split() for r in ref_list]
            
            if not tokenized_pred or not any(tokenized_refs):
                score = 0.0 
            else:
                score = sentence_bleu(tokenized_refs, tokenized_pred, smoothing_function=smoothie)
                
            total_bleu += score
            count += 1
        except Exception as e:
            logging.getLogger("code_generation").warning(f"Could not compute BLEU for pred: '{pred[:50]}...', refs: '{ref_list}'. Error: {e}")
            continue 
            
    if count == 0:
        logging.getLogger("code_generation").warning("BLEU score could not be computed for any pairs.")
        return 0.0
        
    return total_bleu / count

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute average Exact Match (EM) score.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Average Exact Match score (case-sensitive, ignores leading/trailing whitespace).
    """
    if len(predictions) != len(references):
        logging.getLogger("code_generation").error(f"EM: Preds ({len(predictions)}) and refs ({len(references)}) length mismatch.")
        return 0.0
    if not predictions: 
        return 0.0

    match_count = 0
    for pred, ref in zip(predictions, references):
        if str(pred).strip() == str(ref).strip():
            match_count += 1
            
    return match_count / len(predictions)

def plot_training_metrics(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]], 
    val_metrics: Dict[str, List[float]],
    lr_history: List[float],
    output_dir: Path
) -> None:
    """Plot and save training loss, validation metrics, and learning rate."""
    if not train_losses:
        logging.getLogger("code_generation").warning("No training loss data to plot.")
        return
        
    epochs = range(len(train_losses))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics')

    axes[0, 0].plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    if 'bleu' in val_metrics and val_metrics['bleu']:
        axes[0, 1].plot(epochs, val_metrics['bleu'], label='Val BLEU', color='orange')
    axes[0, 1].set_title('Validation BLEU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if 'exact_match' in val_metrics and val_metrics['exact_match']:
        axes[1, 0].plot(epochs, val_metrics['exact_match'], label='Val Exact Match', color='orange')
    axes[1, 0].set_title('Validation Exact Match')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if lr_history:
        axes[1, 1].plot(epochs, lr_history, label='Learning Rate', color='green')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plot_filename = f'training_metrics_{timestamp}.png'
    plot_path = output_dir / plot_filename
    try:
        plt.savefig(plot_path)
        logging.getLogger("code_generation").info(f"Training metrics plot saved to {plot_path}")
    except Exception as e:
        logging.getLogger("code_generation").error(f"Failed to save plot: {e}")
    plt.close()

def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

def load_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Path
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary into a string."""
    return ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())

def get_gpu_memory_usage() -> float:
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def code_clean(code: str) -> str:
    """Clean and format code string."""
    code = code.replace('<code>', '').replace('</code>', '')
    code = code.replace('<python>', '').replace('</python>', '')
    
    code = code.strip()
    
    return code

def create_progress_bar(total: int, current: int, bar_length: int = 50) -> str:
    """Create a progress bar string."""
    progress = float(current) / total
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    return f'[{bar}] {int(progress * 100)}%'
