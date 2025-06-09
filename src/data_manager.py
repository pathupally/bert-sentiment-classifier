"""
Data management classes for BERT Sentiment Classifier.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASET_NAME, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, DATA_DIR
from utils import setup_logging

logger = logging.getLogger(__name__)


class BaseDataManager(ABC):
    """Abstract base class for data management."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory to store data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Initializing {self.__class__.__name__} with data_dir: {data_dir}")
    
    @abstractmethod
    def load_data(self) -> DatasetDict:
        """Load the dataset."""
        pass
    
    @abstractmethod
    def split_data(self, dataset: DatasetDict) -> Dict[str, Dataset]:
        """Split dataset into train/val/test."""
        pass
    
    @abstractmethod
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset."""
        pass
    
    def save_data(self, dataset: Dataset, filename: str) -> None:
        """Save dataset to file."""
        filepath = os.path.join(self.data_dir, filename)
        dataset.save_to_disk(filepath)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_saved_data(self, filename: str) -> Dataset:
        """Load dataset from saved file."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        dataset = Dataset.load_from_disk(filepath)
        logger.info(f"Dataset loaded from {filepath}")
        return dataset


class IMDBDataManager(BaseDataManager):
    """Data manager for IMDB sentiment dataset."""
    
    def __init__(self, dataset_name: str = DATASET_NAME, data_dir: str = DATA_DIR):
        """
        Initialize IMDB data manager.
        
        Args:
            dataset_name: Name of the dataset to load
            data_dir: Directory to store data
        """
        super().__init__(data_dir)
        self.dataset_name = dataset_name
        self.dataset = None
        self.splits = {}
        
        logger.info(f"IMDB data manager initialized for {dataset_name}")
    
    def load_data(self) -> DatasetDict:
        """
        Load IMDB dataset from Hugging Face.
        
        Returns:
            Dataset dictionary containing train and test splits
        """
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        try:
            self.dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset loaded successfully")
            logger.info(f"  Train: {len(self.dataset['train'])} samples")
            logger.info(f"  Test: {len(self.dataset['test'])} samples")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def split_data(self, dataset: DatasetDict = None) -> Dict[str, Dataset]:
        """
        Split the training data into train/validation/test sets.
        
        Args:
            dataset: Dataset to split (uses loaded dataset if None)
            
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        if dataset is None:
            dataset = self.dataset or self.load_data()
        
        logger.info("Splitting dataset into train/validation/test...")
        
        # Split the training data
        train_val_test = dataset['train'].train_test_split(
            test_size=VAL_SPLIT + TEST_SPLIT, 
            seed=42
        )
        
        # Split the remaining data into validation and test
        val_test = train_val_test['test'].train_test_split(
            test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
            seed=42
        )
        
        self.splits = {
            'train': train_val_test['train'],
            'validation': val_test['train'],
            'test': dataset['test']  # Use original test set
        }
        
        logger.info("Dataset split completed:")
        logger.info(f"  Train: {len(self.splits['train'])} samples")
        logger.info(f"  Validation: {len(self.splits['validation'])} samples")
        logger.info(f"  Test: {len(self.splits['test'])} samples")
        
        return self.splits
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset (basic cleaning).
        
        Args:
            dataset: Dataset to preprocess
            
        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset...")
        
        def clean_text(example):
            """Clean text by removing extra whitespace."""
            example['text'] = ' '.join(example['text'].split())
            return example
        
        # Apply text cleaning
        processed_dataset = dataset.map(clean_text)
        
        logger.info("Dataset preprocessing completed")
        return processed_dataset
    
    def get_sample_data(self, split: str = 'train', n_samples: int = 5) -> List[Dict]:
        """
        Get sample data from a specific split.
        
        Args:
            split: Dataset split to sample from
            n_samples: Number of samples to return
            
        Returns:
            List of sample data
        """
        if not self.splits:
            self.split_data()
        
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.splits.keys())}")
        
        dataset = self.splits[split]
        samples = dataset.select(range(min(n_samples, len(dataset))))
        
        return samples.to_list()
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.splits:
            self.split_data()
        
        stats = {}
        
        for split_name, dataset in self.splits.items():
            # Text length statistics
            text_lengths = [len(text.split()) for text in dataset['text']]
            
            stats[split_name] = {
                'num_samples': len(dataset),
                'avg_text_length': sum(text_lengths) / len(text_lengths),
                'min_text_length': min(text_lengths),
                'max_text_length': max(text_lengths),
                'label_distribution': dataset['label'].count()
            }
        
        return stats
    
    def save_splits(self) -> None:
        """Save all dataset splits to disk."""
        if not self.splits:
            self.split_data()
        
        for split_name, dataset in self.splits.items():
            filename = f"{self.dataset_name}_{split_name}.hf"
            self.save_data(dataset, filename)
    
    def load_splits(self) -> Dict[str, Dataset]:
        """Load all dataset splits from disk."""
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            filename = f"{self.dataset_name}_{split_name}.hf"
            try:
                splits[split_name] = self.load_saved_data(filename)
            except FileNotFoundError:
                logger.warning(f"Split file not found: {filename}")
        
        if splits:
            self.splits = splits
            logger.info("Dataset splits loaded from disk")
        
        return splits


class CustomDataManager(BaseDataManager):
    """Data manager for custom datasets."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        """Initialize custom data manager."""
        super().__init__(data_dir)
        self.dataset = None
    
    def load_from_csv(self, filepath: str, text_column: str = 'text', 
                     label_column: str = 'label') -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset from CSV: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")
        
        # Convert to dataset
        dataset = Dataset.from_pandas(df)
        
        # Rename columns if needed
        if text_column != 'text':
            dataset = dataset.rename_column(text_column, 'text')
        if label_column != 'label':
            dataset = dataset.rename_column(label_column, 'label')
        
        self.dataset = dataset
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        return dataset
    
    def load_from_json(self, filepath: str, text_column: str = 'text',
                      label_column: str = 'label') -> Dataset:
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to JSON file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset from JSON: {filepath}")
        
        df = pd.read_json(filepath)
        
        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in JSON")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in JSON")
        
        # Convert to dataset
        dataset = Dataset.from_pandas(df)
        
        # Rename columns if needed
        if text_column != 'text':
            dataset = dataset.rename_column(text_column, 'text')
        if label_column != 'label':
            dataset = dataset.rename_column(label_column, 'label')
        
        self.dataset = dataset
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        return dataset
    
    def load_data(self) -> DatasetDict:
        """Load data (placeholder for custom implementation)."""
        raise NotImplementedError("Use load_from_csv or load_from_json for custom data")
    
    def split_data(self, dataset: DatasetDict = None) -> Dict[str, Dataset]:
        """Split dataset into train/val/test."""
        if dataset is None:
            dataset = self.dataset
        
        if dataset is None:
            raise ValueError("No dataset loaded")
        
        # Split into train/val/test
        train_val, test = train_test_split(
            dataset, test_size=TEST_SPLIT, random_state=42, stratify=dataset['label']
        )
        
        train, val = train_test_split(
            train_val, test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=42, stratify=train_val['label']
        )
        
        self.splits = {
            'train': train,
            'validation': val,
            'test': test
        }
        
        return self.splits
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset."""
        return dataset  # No preprocessing by default for custom data


class DataManagerFactory:
    """Factory class for creating data managers."""
    
    @staticmethod
    def create_data_manager(data_type: str = "imdb", **kwargs) -> BaseDataManager:
        """
        Create a data manager instance.
        
        Args:
            data_type: Type of data manager to create
            **kwargs: Additional arguments for data manager creation
            
        Returns:
            Data manager instance
        """
        if data_type.lower() == "imdb":
            return IMDBDataManager(**kwargs)
        elif data_type.lower() == "custom":
            return CustomDataManager(**kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    @staticmethod
    def get_available_data_types() -> List[str]:
        """Get list of available data types."""
        return ["imdb", "custom"] 