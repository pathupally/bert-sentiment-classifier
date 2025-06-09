"""
Utility functions for the BERT Sentiment Classifier project.
"""
import os
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import MAX_LENGTH, SENTIMENT_LABELS, OUTPUTS_DIR, RANDOM_SEED

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def tokenize_function(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize text examples using the provided tokenizer.
    
    Args:
        examples: Dictionary containing text data
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Dictionary with tokenized inputs
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Prepare dataset by tokenizing and formatting for training.
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Processed dataset ready for training
    """
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Rename columns to match expected format
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def save_evaluation_results(metrics: Dict[str, float], 
                          save_path: Optional[str] = None) -> None:
    """
    Save evaluation results to a text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Path to save the results (optional)
    """
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "evaluation_results.txt")
    
    with open(save_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
    
    logger.info(f"Evaluation results saved to {save_path}")


def format_prediction_output(prediction: Dict) -> Dict:
    """
    Format prediction output for display.
    
    Args:
        prediction: Raw prediction from model
        
    Returns:
        Formatted prediction with sentiment and confidence
    """
    probabilities = torch.softmax(torch.tensor(prediction['logits']), dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    return {
        'sentiment': SENTIMENT_LABELS[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0].item(),
            'positive': probabilities[1].item()
        }
    }


def validate_text_input(text: str) -> bool:
    """
    Validate text input for prediction.
    
    Args:
        text: Input text to validate
        
    Returns:
        True if text is valid, False otherwise
    """
    if not text or not text.strip():
        return False
    
    if len(text.strip()) < 3:
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean and preprocess text input.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic cleaning (can be extended)
    text = text.strip()
    
    return text


def get_device_info() -> str:
    """
    Get information about the current device.
    
    Returns:
        String describing the current device
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA: {device_name} ({memory:.1f}GB)"
    else:
        return "CPU"


def log_training_info(config: Dict) -> None:
    """
    Log training configuration information.
    
    Args:
        config: Training configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("Training Configuration")
    logger.info("=" * 50)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"Device: {get_device_info()}")
    logger.info("=" * 50) 