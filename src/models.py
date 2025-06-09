"""
Model classes for BERT Sentiment Classifier.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

from config import (
    MODEL_NAME, MAX_LENGTH, NUM_LABELS, SENTIMENT_LABELS,
    TRAINING_ARGS, MODEL_PATH, DEVICE
)
from utils import setup_logging, get_device_info

logger = logging.getLogger(__name__)


class BaseSentimentModel(ABC):
    """Abstract base class for sentiment analysis models."""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model to load
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or DEVICE
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
        logger.info(f"Initializing {self.__class__.__name__} with {model_name}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction on a single text."""
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on a batch of texts."""
        pass
    
    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.device = device
            self.model = self.model.to(device)
            logger.info(f"Model moved to {device}")
    
    def save_model(self, path: str) -> None:
        """Save the model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_from_path(self, path: str) -> None:
        """Load model and tokenizer from a saved path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.is_trained = True
        
        # Move to device
        self.to_device(self.device)
        
        logger.info(f"Model loaded from {path}")


class BERTSentimentModel(BaseSentimentModel):
    """BERT-based sentiment analysis model."""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """Initialize BERT sentiment model."""
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self) -> None:
        """Load BERT model and tokenizer."""
        logger.info(f"Loading BERT model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            problem_type="single_label_classification"
        )
        
        # Move to device
        self.to_device(self.device)
        
        logger.info(f"BERT model loaded successfully")
        logger.info(f"  Model parameters: {self.model.num_parameters():,}")
        logger.info(f"  Device: {self.device}")
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text."""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction on a single text."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenize_text(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        
        # Format result
        result = {
            'text': text,
            'sentiment': SENTIMENT_LABELS[prediction.item()],
            'confidence': probabilities[0][prediction.item()].item(),
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'positive': probabilities[0][1].item()
            },
            'logits': logits.cpu().numpy()
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on a batch of texts."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if not texts:
            return []
        
        # Tokenize inputs
        inputs = self.tokenize_batch(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Format results
        results = []
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            result = {
                'text': text,
                'sentiment': SENTIMENT_LABELS[pred.item()],
                'confidence': probs[pred].item(),
                'probabilities': {
                    'negative': probs[0].item(),
                    'positive': probs[1].item()
                },
                'logits': logits[i].cpu().numpy()
            }
            results.append(result)
        
        return results


class ModelTrainer:
    """Class for training sentiment analysis models."""
    
    def __init__(self, model: BERTSentimentModel, training_args: Dict = None):
        """
        Initialize the trainer.
        
        Args:
            model: BERT sentiment model to train
            training_args: Training arguments dictionary
        """
        self.model = model
        self.training_args = training_args or TRAINING_ARGS
        self.trainer = None
        self.training_history = []
        
        logger.info("Model trainer initialized")
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training."""
        from utils import tokenize_function
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, self.model.tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Rename columns to match expected format
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Trained Trainer object
        """
        from utils import compute_metrics
        
        logger.info("Preparing datasets for training...")
        
        # Prepare datasets
        train_dataset_processed = self.prepare_dataset(train_dataset)
        val_dataset_processed = self.prepare_dataset(val_dataset)
        
        # Set up training arguments
        training_args = TrainingArguments(**self.training_args)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            tokenizer=self.model.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Store training history
        self.training_history = train_result
        
        # Mark model as trained
        self.model.is_trained = True
        
        logger.info("Training completed successfully!")
        
        return self.trainer
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        test_dataset_processed = self.prepare_dataset(test_dataset)
        results = self.trainer.evaluate(test_dataset_processed)
        
        return results
    
    def save_trained_model(self, path: str = None) -> None:
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError("No trained model to save")
        
        path = path or MODEL_PATH
        self.trainer.save_model()
        self.model.tokenizer.save_pretrained(path)
        
        logger.info(f"Trained model saved to {path}")


class ModelManager:
    """Manager class for handling multiple models and model operations."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.current_model = None
        
        logger.info("Model manager initialized")
    
    def create_model(self, name: str, model_type: str = "bert", **kwargs) -> BaseSentimentModel:
        """
        Create a new model instance.
        
        Args:
            name: Name for the model
            model_type: Type of model to create
            **kwargs: Additional arguments for model creation
            
        Returns:
            Created model instance
        """
        if model_type.lower() == "bert":
            model = BERTSentimentModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.models[name] = model
        
        if self.current_model is None:
            self.current_model = name
        
        logger.info(f"Created {model_type} model: {name}")
        return model
    
    def load_model(self, name: str, path: str) -> BaseSentimentModel:
        """
        Load a model from a saved path.
        
        Args:
            name: Name for the model
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model = BERTSentimentModel()
        model.load_from_path(path)
        
        self.models[name] = model
        
        if self.current_model is None:
            self.current_model = name
        
        logger.info(f"Loaded model from {path}: {name}")
        return model
    
    def get_model(self, name: str = None) -> BaseSentimentModel:
        """
        Get a model by name.
        
        Args:
            name: Name of the model (uses current if None)
            
        Returns:
            Model instance
        """
        model_name = name or self.current_model
        
        if model_name is None:
            raise ValueError("No model available")
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return self.models[model_name]
    
    def set_current_model(self, name: str) -> None:
        """Set the current model."""
        if name not in self.models:
            raise ValueError(f"Model not found: {name}")
        
        self.current_model = name
        logger.info(f"Current model set to: {name}")
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            
            if self.current_model == name:
                self.current_model = None
                if self.models:
                    self.current_model = list(self.models.keys())[0]
            
            logger.info(f"Removed model: {name}")
        else:
            logger.warning(f"Model not found: {name}")
    
    def predict_with_current(self, text: str) -> Dict[str, Any]:
        """Make prediction using the current model."""
        model = self.get_model()
        return model.predict(text)
    
    def predict_batch_with_current(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make batch predictions using the current model."""
        model = self.get_model()
        return model.predict_batch(texts) 