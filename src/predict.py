#!/usr/bin/env python3
"""
Prediction script for BERT Sentiment Classifier.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Union

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL
from utils import setup_logging, validate_text_input, clean_text
from models import ModelManager, BERTSentimentModel

# Set up logging
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


class SentimentPredictionPipeline:
    """Complete prediction pipeline for sentiment analysis."""
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        self.model_manager = ModelManager()
        self.current_model = None
        
        logger.info("Sentiment prediction pipeline initialized")
    
    def load_model(self, model_path: str = None, model_name: str = "prediction_model") -> None:
        """
        Load a trained model for prediction.
        
        Args:
            model_path: Path to the trained model
            model_name: Name for the model in the manager
        """
        logger.info("Loading model for prediction...")
        
        if model_path is None:
            from config import MODEL_PATH
            model_path = MODEL_PATH
        
        # Load the model
        self.model_manager.load_model(model_name, model_path)
        self.model_manager.set_current_model(model_name)
        self.current_model = self.model_manager.get_model()
        
        logger.info("Model loaded successfully for prediction")
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing prediction results
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Validate and clean input
        if not validate_text_input(text):
            raise ValueError("Invalid text input. Text must be non-empty and at least 3 characters long.")
        
        text = clean_text(text)
        
        # Make prediction
        result = self.current_model.predict(text)
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of dictionaries containing prediction results
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if not texts:
            return []
        
        # Clean and validate all texts
        cleaned_texts = []
        for text in texts:
            if validate_text_input(text):
                cleaned_texts.append(clean_text(text))
            else:
                logger.warning(f"Skipping invalid text: '{text[:50]}...'")
        
        if not cleaned_texts:
            raise ValueError("No valid texts found in the input list")
        
        # Make predictions
        results = self.current_model.predict_batch(cleaned_texts)
        
        return results
    
    def run_prediction(self, text: str = None, texts: List[str] = None) -> Union[Dict, List[Dict]]:
        """
        Run prediction on text or texts.
        
        Args:
            text: Single text to predict (optional)
            texts: List of texts to predict (optional)
            
        Returns:
            Prediction results
        """
        if text is not None:
            return self.predict_single(text)
        elif texts is not None:
            return self.predict_batch(texts)
        else:
            raise ValueError("Either 'text' or 'texts' must be provided")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.current_model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.current_model.model_name,
            "device": self.current_model.device,
            "is_trained": self.current_model.is_trained,
            "model_type": type(self.current_model).__name__
        }


# Backward compatibility functions
def predict_sentiment(text: str, model_path: str = None) -> Dict:
    """
    Convenience function for single text prediction (backward compatibility).
    
    Args:
        text: Input text to analyze
        model_path: Path to the trained model (optional)
        
    Returns:
        Dictionary containing prediction results
    """
    pipeline = SentimentPredictionPipeline()
    pipeline.load_model(model_path)
    return pipeline.predict_single(text)


def predict_sentiments(texts: List[str], model_path: str = None) -> List[Dict]:
    """
    Convenience function for batch text prediction (backward compatibility).
    
    Args:
        texts: List of input texts to analyze
        model_path: Path to the trained model (optional)
        
    Returns:
        List of dictionaries containing prediction results
    """
    pipeline = SentimentPredictionPipeline()
    pipeline.load_model(model_path)
    return pipeline.predict_batch(texts)


# Legacy SentimentPredictor class for backward compatibility
class SentimentPredictor:
    """Legacy class for making sentiment predictions (backward compatibility)."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_path: Path to the trained model (optional)
        """
        self.pipeline = SentimentPredictionPipeline()
        self.pipeline.load_model(model_path)
    
    def predict_single(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        return self.pipeline.predict_single(text)
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a batch of texts."""
        return self.pipeline.predict_batch(texts)


def print_prediction_result(result: Dict):
    """
    Print prediction result in a formatted way.
    
    Args:
        result: Prediction result dictionary
    """
    print("=" * 50)
    print("Sentiment Analysis Result")
    print("=" * 50)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nProbabilities:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  {sentiment.capitalize()}: {prob:.4f}")
    print("=" * 50)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="BERT Sentiment Classifier Prediction")
    parser.add_argument("text", nargs="?", help="Text to analyze for sentiment")
    parser.add_argument("--model-path", help="Path to the trained model")
    parser.add_argument("--batch", action="store_true", help="Process multiple texts (one per line from stdin)")
    parser.add_argument("--model-name", default="prediction_model", help="Name for the model")
    
    args = parser.parse_args()
    
    # Create prediction pipeline
    pipeline = SentimentPredictionPipeline()
    
    try:
        # Load model
        pipeline.load_model(args.model_path, args.model_name)
        
        # Print model info
        model_info = pipeline.get_model_info()
        print(f"Model loaded: {model_info['model_name']} on {model_info['device']}")
        
        if args.batch:
            # Batch mode: read from stdin
            print("Enter texts (one per line, Ctrl+D to finish):")
            texts = []
            try:
                while True:
                    line = input()
                    if line.strip():
                        texts.append(line.strip())
            except EOFError:
                pass
            
            if not texts:
                print("No texts provided")
                return
            
            results = pipeline.predict_batch(texts)
            
            print(f"\nProcessed {len(results)} texts:")
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['text'][:50]}...")
                print(f"   Sentiment: {result['sentiment'].upper()} (Confidence: {result['confidence']:.3f})")
        
        elif args.text:
            # Single text mode
            result = pipeline.predict_single(args.text)
            print_prediction_result(result)
        
        else:
            # Interactive mode
            print("BERT Sentiment Classifier - Interactive Mode")
            print("Enter text to analyze (or 'quit' to exit):")
            
            while True:
                try:
                    text = input("\n> ")
                    if text.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if text.strip():
                        result = pipeline.predict_single(text)
                        print_prediction_result(result)
                    else:
                        print("Please enter some text to analyze.")
                
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 