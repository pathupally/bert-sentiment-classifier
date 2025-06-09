#!/usr/bin/env python3
"""
Training script for BERT Sentiment Classifier.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL
from utils import setup_logging, log_training_info, get_device_info
from models import BERTSentimentModel, ModelTrainer, ModelManager
from data_manager import IMDBDataManager, DataManagerFactory
from evaluator import SentimentEvaluator, EvaluatorFactory

# Set up logging
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


class SentimentTrainingPipeline:
    """Complete training pipeline for sentiment analysis."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data_manager = None
        self.model_manager = ModelManager()
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.training_results = {}
        
        logger.info("Sentiment training pipeline initialized")
    
    def setup_data(self, data_type: str = "imdb") -> Dict[str, Any]:
        """
        Set up data loading and preprocessing.
        
        Args:
            data_type: Type of data to load
            
        Returns:
            Dictionary containing dataset splits
        """
        logger.info(f"Setting up {data_type} data...")
        
        # Create data manager
        self.data_manager = DataManagerFactory.create_data_manager(data_type)
        
        # Load and split data
        dataset = self.data_manager.load_data()
        splits = self.data_manager.split_data(dataset)
        
        # Preprocess data
        for split_name, split_data in splits.items():
            splits[split_name] = self.data_manager.preprocess_data(split_data)
        
        logger.info("Data setup completed")
        return splits
    
    def setup_model(self, model_name: str = "bert_sentiment") -> None:
        """
        Set up the model for training.
        
        Args:
            model_name: Name for the model
        """
        logger.info("Setting up model...")
        
        # Create BERT model
        self.model = self.model_manager.create_model(
            name=model_name,
            model_type="bert"
        )
        
        logger.info("Model setup completed")
    
    def setup_trainer(self) -> None:
        """Set up the model trainer."""
        logger.info("Setting up trainer...")
        
        self.trainer = ModelTrainer(self.model)
        logger.info("Trainer setup completed")
    
    def setup_evaluator(self) -> None:
        """Set up the model evaluator."""
        logger.info("Setting up evaluator...")
        
        self.evaluator = EvaluatorFactory.create_evaluator(
            evaluator_type="sentiment",
            model=self.model
        )
        logger.info("Evaluator setup completed")
    
    def train_model(self, train_dataset, val_dataset) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training results
        """
        logger.info("Starting model training...")
        
        # Train the model
        trainer_result = self.trainer.train(train_dataset, val_dataset)
        
        # Save the trained model
        self.trainer.save_trained_model()
        
        # Store training results
        self.training_results = {
            'trainer_result': trainer_result,
            'model_path': self.model.model_path,
            'is_trained': self.model.is_trained
        }
        
        logger.info("Model training completed")
        return self.training_results
    
    def evaluate_model(self, test_dataset) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating trained model...")
        
        # Evaluate the model
        evaluation_results = self.evaluator.evaluate(test_dataset)
        
        # Create visualizations
        saved_plots = self.evaluator.create_visualizations(evaluation_results)
        
        # Save evaluation report
        report_path = self.evaluator.save_evaluation_report(evaluation_results)
        
        # Store evaluation results
        evaluation_summary = {
            'results': evaluation_results,
            'saved_plots': saved_plots,
            'report_path': report_path,
            'summary': self.evaluator.get_evaluation_summary()
        }
        
        logger.info("Model evaluation completed")
        return evaluation_summary
    
    def run_complete_pipeline(self, data_type: str = "imdb", 
                            model_name: str = "bert_sentiment") -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_type: Type of data to use
            model_name: Name for the model
            
        Returns:
            Complete pipeline results
        """
        logger.info("=" * 60)
        logger.info("BERT Sentiment Classifier - Complete Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup data
            splits = self.setup_data(data_type)
            
            # Step 2: Setup model
            self.setup_model(model_name)
            
            # Step 3: Setup trainer
            self.setup_trainer()
            
            # Step 4: Setup evaluator
            self.setup_evaluator()
            
            # Step 5: Train model
            training_results = self.train_model(splits['train'], splits['validation'])
            
            # Step 6: Evaluate model
            evaluation_results = self.evaluate_model(splits['test'])
            
            # Compile complete results
            pipeline_results = {
                'training': training_results,
                'evaluation': evaluation_results,
                'data_stats': self.data_manager.get_data_stats(),
                'model_info': {
                    'model_name': model_name,
                    'data_type': data_type,
                    'device': get_device_info()
                }
            }
            
            logger.info("=" * 60)
            logger.info("Complete training pipeline finished successfully!")
            logger.info("=" * 60)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the training pipeline."""
        summary = {
            'model_manager': {
                'available_models': self.model_manager.list_models(),
                'current_model': self.model_manager.current_model
            },
            'training_results': self.training_results,
            'data_manager': {
                'data_type': type(self.data_manager).__name__ if self.data_manager else None
            }
        }
        
        if self.evaluator:
            summary['evaluation_summary'] = self.evaluator.get_evaluation_summary()
        
        return summary


def main():
    """Main function for command-line training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT Sentiment Classifier Training")
    parser.add_argument("--data-type", default="imdb", help="Type of data to use")
    parser.add_argument("--model-name", default="bert_sentiment", help="Name for the model")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with small dataset")
    
    args = parser.parse_args()
    
    # Create and run training pipeline
    pipeline = SentimentTrainingPipeline()
    
    try:
        results = pipeline.run_complete_pipeline(
            data_type=args.data_type,
            model_name=args.model_name
        )
        
        # Print summary
        print("\nTraining Pipeline Summary:")
        print("=" * 40)
        print(f"Model: {results['model_info']['model_name']}")
        print(f"Data: {results['model_info']['data_type']}")
        print(f"Device: {results['model_info']['device']}")
        
        if 'evaluation' in results:
            eval_results = results['evaluation']['results']
            print(f"\nEvaluation Results:")
            print(f"  Accuracy: {eval_results['overall_metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {eval_results['overall_metrics']['f1_score']:.4f}")
            print(f"  Precision: {eval_results['overall_metrics']['precision']:.4f}")
            print(f"  Recall: {eval_results['overall_metrics']['recall']:.4f}")
        
        print(f"\nResults saved to: {results['evaluation']['report_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 