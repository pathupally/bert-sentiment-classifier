#!/usr/bin/env python3
"""
Evaluation script for BERT Sentiment Classifier.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL
from utils import setup_logging, get_device_info
from models import ModelManager
from data_manager import DataManagerFactory
from evaluator import EvaluatorFactory

# Set up logging
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


class SentimentEvaluationPipeline:
    """Complete evaluation pipeline for sentiment analysis."""
    
    def __init__(self):
        """Initialize the evaluation pipeline."""
        self.model_manager = ModelManager()
        self.data_manager = None
        self.evaluator = None
        self.evaluation_results = {}
        
        logger.info("Sentiment evaluation pipeline initialized")
    
    def load_trained_model(self, model_path: str = None, model_name: str = "trained_model") -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the trained model
            model_name: Name for the model in the manager
        """
        logger.info("Loading trained model...")
        
        if model_path is None:
            from config import MODEL_PATH
            model_path = MODEL_PATH
        
        # Load the model
        self.model_manager.load_model(model_name, model_path)
        self.model_manager.set_current_model(model_name)
        
        logger.info("Trained model loaded successfully")
    
    def setup_data(self, data_type: str = "imdb") -> Dict[str, Any]:
        """
        Set up data for evaluation.
        
        Args:
            data_type: Type of data to load
            
        Returns:
            Dictionary containing dataset splits
        """
        logger.info(f"Setting up {data_type} data for evaluation...")
        
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
    
    def setup_evaluator(self) -> None:
        """Set up the model evaluator."""
        logger.info("Setting up evaluator...")
        
        # Get current model
        model = self.model_manager.get_model()
        
        # Create evaluator
        self.evaluator = EvaluatorFactory.create_evaluator(
            evaluator_type="sentiment",
            model=model
        )
        
        logger.info("Evaluator setup completed")
    
    def evaluate_model(self, test_dataset) -> Dict[str, Any]:
        """
        Evaluate the model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        # Evaluate the model
        evaluation_results = self.evaluator.evaluate(test_dataset)
        
        # Create visualizations
        saved_plots = self.evaluator.create_visualizations(evaluation_results)
        
        # Save evaluation report
        report_path = self.evaluator.save_evaluation_report(evaluation_results)
        
        # Store evaluation results
        self.evaluation_results = {
            'results': evaluation_results,
            'saved_plots': saved_plots,
            'report_path': report_path,
            'summary': self.evaluator.get_evaluation_summary()
        }
        
        logger.info("Model evaluation completed")
        return self.evaluation_results
    
    def run_complete_evaluation(self, model_path: str = None, 
                              data_type: str = "imdb") -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            model_path: Path to the trained model
            data_type: Type of data to use
            
        Returns:
            Complete evaluation results
        """
        logger.info("=" * 60)
        logger.info("BERT Sentiment Classifier - Complete Evaluation Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load trained model
            self.load_trained_model(model_path)
            
            # Step 2: Setup data
            splits = self.setup_data(data_type)
            
            # Step 3: Setup evaluator
            self.setup_evaluator()
            
            # Step 4: Evaluate model
            evaluation_results = self.evaluate_model(splits['test'])
            
            # Compile complete results
            pipeline_results = {
                'evaluation': evaluation_results,
                'data_stats': self.data_manager.get_data_stats(),
                'model_info': {
                    'current_model': self.model_manager.current_model,
                    'available_models': self.model_manager.list_models(),
                    'data_type': data_type,
                    'device': get_device_info()
                }
            }
            
            logger.info("=" * 60)
            logger.info("Complete evaluation pipeline finished successfully!")
            logger.info("=" * 60)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {str(e)}")
            raise
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of the evaluation pipeline."""
        summary = {
            'model_manager': {
                'available_models': self.model_manager.list_models(),
                'current_model': self.model_manager.current_model
            },
            'evaluation_results': self.evaluation_results,
            'data_manager': {
                'data_type': type(self.data_manager).__name__ if self.data_manager else None
            }
        }
        
        if self.evaluator:
            summary['evaluation_summary'] = self.evaluator.get_evaluation_summary()
        
        return summary


def main():
    """Main function for command-line evaluation."""
    parser = argparse.ArgumentParser(description="BERT Sentiment Classifier Evaluation")
    parser.add_argument("--model-path", help="Path to the trained model")
    parser.add_argument("--data-type", default="imdb", help="Type of data to use")
    parser.add_argument("--model-name", default="trained_model", help="Name for the model")
    
    args = parser.parse_args()
    
    # Create and run evaluation pipeline
    pipeline = SentimentEvaluationPipeline()
    
    try:
        results = pipeline.run_complete_evaluation(
            model_path=args.model_path,
            data_type=args.data_type
        )
        
        # Print summary
        print("\nEvaluation Pipeline Summary:")
        print("=" * 40)
        print(f"Model: {results['model_info']['current_model']}")
        print(f"Data: {results['model_info']['data_type']}")
        print(f"Device: {results['model_info']['device']}")
        
        if 'evaluation' in results:
            eval_results = results['evaluation']['results']
            print(f"\nEvaluation Results:")
            print(f"  Accuracy: {eval_results['overall_metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {eval_results['overall_metrics']['f1_score']:.4f}")
            print(f"  Precision: {eval_results['overall_metrics']['precision']:.4f}")
            print(f"  Recall: {eval_results['overall_metrics']['recall']:.4f}")
            
            if eval_results['overall_metrics']['roc_auc'] is not None:
                print(f"  ROC-AUC: {eval_results['overall_metrics']['roc_auc']:.4f}")
        
        print(f"\nResults saved to: {results['evaluation']['report_path']}")
        print(f"Plots saved to: {len(results['evaluation']['saved_plots'])} files")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 