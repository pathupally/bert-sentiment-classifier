"""
Evaluation classes for BERT Sentiment Classifier.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import OUTPUTS_DIR, SENTIMENT_LABELS
from utils import setup_logging, plot_confusion_matrix, save_evaluation_results
from models import BaseSentimentModel

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for model evaluation."""
    
    def __init__(self, model: BaseSentimentModel, outputs_dir: str = OUTPUTS_DIR):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            outputs_dir: Directory to save evaluation results
        """
        self.model = model
        self.outputs_dir = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)
        
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def evaluate(self, test_dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the model on test dataset."""
        pass
    
    @abstractmethod
    def generate_reports(self, predictions: List[int], 
                        true_labels: List[int], 
                        probabilities: np.ndarray) -> Dict[str, Any]:
        """Generate evaluation reports."""
        pass
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to file."""
        filepath = os.path.join(self.outputs_dir, filename)
        
        if filename.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Evaluation results saved to {filepath}")


class SentimentEvaluator(BaseEvaluator):
    """Evaluator for sentiment analysis models."""
    
    def __init__(self, model: BaseSentimentModel, outputs_dir: str = OUTPUTS_DIR):
        """Initialize sentiment evaluator."""
        super().__init__(model, outputs_dir)
        self.evaluation_history = []
    
    def predict_dataset(self, dataset: Dataset, batch_size: int = 32) -> Tuple[List[int], np.ndarray]:
        """
        Make predictions on entire dataset.
        
        Args:
            dataset: Dataset to predict on
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info(f"Making predictions on {len(dataset)} samples...")
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset['text'][i:i + batch_size]
            batch_results = self.model.predict_batch(batch_texts)
            
            for result in batch_results:
                pred_label = 0 if result['sentiment'] == 'negative' else 1
                all_predictions.append(pred_label)
                all_probabilities.append([result['probabilities']['negative'], 
                                       result['probabilities']['positive']])
        
        return all_predictions, np.array(all_probabilities)
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate the model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting model evaluation...")
        
        # Get true labels
        true_labels = test_dataset['label']
        
        # Make predictions
        predictions, probabilities = self.predict_dataset(test_dataset)
        
        # Generate reports
        results = self.generate_reports(predictions, true_labels, probabilities)
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results
        })
        
        logger.info("Evaluation completed")
        return results
    
    def generate_reports(self, predictions: List[int], 
                        true_labels: List[int], 
                        probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation reports.
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary containing all evaluation metrics and reports
        """
        logger.info("Generating evaluation reports...")
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # ROC-AUC (for binary classification)
        if len(np.unique(true_labels)) == 2:
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            roc_auc = None
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=SENTIMENT_LABELS, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Compile results
        results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist(),
                'support': support.tolist()
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities.tolist()
        }
        
        return results
    
    def create_visualizations(self, results: Dict[str, Any], 
                            save_plots: bool = True) -> List[str]:
        """
        Create evaluation visualizations.
        
        Args:
            results: Evaluation results
            save_plots: Whether to save plots to disk
            
        Returns:
            List of saved plot filepaths
        """
        logger.info("Creating evaluation visualizations...")
        
        saved_plots = []
        
        # 1. Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        cm_path = os.path.join(self.outputs_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            results['true_labels'], 
            results['predictions'], 
            cm_path
        )
        saved_plots.append(cm_path)
        
        # 2. ROC Curve (if binary classification)
        if results['overall_metrics']['roc_auc'] is not None:
            roc_path = os.path.join(self.outputs_dir, "roc_curve.png")
            self._plot_roc_curve(
                results['true_labels'],
                results['probabilities'][:, 1],
                roc_path
            )
            saved_plots.append(roc_path)
        
        # 3. Metrics Comparison
        metrics_path = os.path.join(self.outputs_dir, "metrics_comparison.png")
        self._plot_metrics_comparison(results, metrics_path)
        saved_plots.append(metrics_path)
        
        # 4. Confidence Distribution
        conf_path = os.path.join(self.outputs_dir, "confidence_distribution.png")
        self._plot_confidence_distribution(results, conf_path)
        saved_plots.append(conf_path)
        
        logger.info(f"Created {len(saved_plots)} visualization plots")
        return saved_plots
    
    def _plot_roc_curve(self, true_labels: List[int], 
                       probabilities: np.ndarray, 
                       save_path: str) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc_score = roc_auc_score(true_labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, results: Dict[str, Any], 
                               save_path: str) -> None:
        """Plot metrics comparison."""
        metrics = results['overall_metrics']
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall metrics
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score']]
        
        bars1 = ax1.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax1.set_title('Overall Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars1, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Per-class metrics
        per_class = results['per_class_metrics']
        x = np.arange(len(SENTIMENT_LABELS))
        width = 0.25
        
        bars2 = ax2.bar(x - width, per_class['precision'], width, label='Precision', color='#3498db')
        bars3 = ax2.bar(x, per_class['recall'], width, label='Recall', color='#2ecc71')
        bars4 = ax2.bar(x + width, per_class['f1_score'], width, label='F1-Score', color='#f39c12')
        
        ax2.set_title('Per-Class Metrics')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Sentiment Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(SENTIMENT_LABELS)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, results: Dict[str, Any], 
                                    save_path: str) -> None:
        """Plot confidence distribution."""
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        probabilities = np.array(results['probabilities'])
        
        # Get confidence scores
        confidences = np.max(probabilities, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = predictions == true_labels
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        if len(correct_confidences) > 0:
            plt.hist(correct_confidences, bins=50, alpha=0.7, 
                    label='Correct Predictions', density=True, color='green')
        if len(incorrect_confidences) > 0:
            plt.hist(incorrect_confidences, bins=50, alpha=0.7, 
                    label='Incorrect Predictions', density=True, color='red')
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Distribution of Prediction Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Save comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Path to saved report
        """
        report_path = os.path.join(self.outputs_dir, "evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BERT Sentiment Classifier - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in results['overall_metrics'].items():
                if value is not None:
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 20 + "\n")
            for i, label in enumerate(SENTIMENT_LABELS):
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  Precision: {results['per_class_metrics']['precision'][i]:.4f}\n")
                f.write(f"  Recall: {results['per_class_metrics']['recall'][i]:.4f}\n")
                f.write(f"  F1-Score: {results['per_class_metrics']['f1_score'][i]:.4f}\n")
                f.write(f"  Support: {results['per_class_metrics']['support'][i]}\n")
            
            f.write("\n")
            
            # Classification report
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(
                results['true_labels'], 
                results['predictions'], 
                target_names=SENTIMENT_LABELS
            ))
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.evaluation_history:
            return {}
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'latest_evaluation': self.evaluation_history[-1],
            'best_accuracy': max([eval['results']['overall_metrics']['accuracy'] 
                                for eval in self.evaluation_history]),
            'average_accuracy': np.mean([eval['results']['overall_metrics']['accuracy'] 
                                       for eval in self.evaluation_history])
        }
        
        return summary


class EvaluatorFactory:
    """Factory class for creating evaluators."""
    
    @staticmethod
    def create_evaluator(evaluator_type: str = "sentiment", **kwargs) -> BaseEvaluator:
        """
        Create an evaluator instance.
        
        Args:
            evaluator_type: Type of evaluator to create
            **kwargs: Additional arguments for evaluator creation
            
        Returns:
            Evaluator instance
        """
        if evaluator_type.lower() == "sentiment":
            return SentimentEvaluator(**kwargs)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    @staticmethod
    def get_available_evaluator_types() -> List[str]:
        """Get list of available evaluator types."""
        return ["sentiment"] 