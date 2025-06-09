#!/usr/bin/env python3
"""
Test script to verify the project setup and dependencies.
Includes tests for the new object-oriented architecture.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print(f"✅ Seaborn: {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn: {e}")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError as e:
        print(f"❌ Gradio: {e}")
        return False
    
    try:
        import dotenv
        print(f"✅ Python-dotenv: {dotenv.__version__}")
    except ImportError as e:
        print(f"❌ Python-dotenv: {e}")
        return False
    
    return True

def test_project_imports():
    """Test that project modules can be imported."""
    print("\nTesting project imports...")
    
    try:
        from src.config import MODEL_NAME, SENTIMENT_LABELS
        print(f"✅ Config: MODEL_NAME={MODEL_NAME}, LABELS={SENTIMENT_LABELS}")
    except ImportError as e:
        print(f"❌ Config: {e}")
        return False
    
    try:
        from src.utils import setup_logging, validate_text_input
        print("✅ Utils: setup_logging, validate_text_input")
    except ImportError as e:
        print(f"❌ Utils: {e}")
        return False
    
    try:
        from src.models import BERTSentimentModel, ModelManager, ModelTrainer
        print("✅ Models: BERTSentimentModel, ModelManager, ModelTrainer")
    except ImportError as e:
        print(f"❌ Models: {e}")
        return False
    
    try:
        from src.data_manager import IMDBDataManager, DataManagerFactory
        print("✅ Data Manager: IMDBDataManager, DataManagerFactory")
    except ImportError as e:
        print(f"❌ Data Manager: {e}")
        return False
    
    try:
        from src.evaluator import SentimentEvaluator, EvaluatorFactory
        print("✅ Evaluator: SentimentEvaluator, EvaluatorFactory")
    except ImportError as e:
        print(f"❌ Evaluator: {e}")
        return False
    
    try:
        from src.predict import SentimentPredictionPipeline
        print("✅ Predict: SentimentPredictionPipeline")
    except ImportError as e:
        print(f"❌ Predict: {e}")
        return False
    
    return True

def test_object_oriented_classes():
    """Test the new object-oriented classes."""
    print("\nTesting object-oriented classes...")
    
    try:
        # Test ModelManager
        from src.models import ModelManager
        model_manager = ModelManager()
        print("✅ ModelManager: Created successfully")
        
        # Test DataManagerFactory
        from src.data_manager import DataManagerFactory
        data_manager = DataManagerFactory.create_data_manager("imdb")
        print("✅ DataManagerFactory: Created IMDB data manager")
        
        # Test EvaluatorFactory
        from src.evaluator import EvaluatorFactory
        print("✅ EvaluatorFactory: Available evaluator types:", 
              EvaluatorFactory.get_available_evaluator_types())
        
        # Test SentimentPredictionPipeline
        from src.predict import SentimentPredictionPipeline
        pipeline = SentimentPredictionPipeline()
        print("✅ SentimentPredictionPipeline: Created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Object-oriented classes: {e}")
        return False

def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = ['src', 'data', 'models', 'outputs']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory: {dir_name}/")
        else:
            print(f"❌ Directory: {dir_name}/ (missing)")
            return False
    
    return True

def test_files():
    """Test that required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        'requirements.txt',
        '.gitignore',
        '.env.example',
        'LICENSE',
        'README.md',
        'src/config.py',
        'src/utils.py',
        'src/models.py',
        'src/data_manager.py',
        'src/evaluator.py',
        'src/train.py',
        'src/evaluate.py',
        'src/predict.py',
        'src/ui.py'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File: {file_name}")
        else:
            print(f"❌ File: {file_name} (missing)")
            return False
    
    return True

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (will use CPU)")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
    
    return True

def test_architecture():
    """Test the object-oriented architecture."""
    print("\nTesting object-oriented architecture...")
    
    try:
        # Test inheritance and abstract classes
        from src.models import BaseSentimentModel, BERTSentimentModel
        from src.data_manager import BaseDataManager, IMDBDataManager
        from src.evaluator import BaseEvaluator, SentimentEvaluator
        
        # Test that concrete classes inherit from abstract base classes
        assert issubclass(BERTSentimentModel, BaseSentimentModel)
        assert issubclass(IMDBDataManager, BaseDataManager)
        assert issubclass(SentimentEvaluator, BaseEvaluator)
        
        print("✅ Inheritance: All classes properly inherit from base classes")
        
        # Test factory patterns
        from src.data_manager import DataManagerFactory
        from src.evaluator import EvaluatorFactory
        
        data_types = DataManagerFactory.get_available_data_types()
        evaluator_types = EvaluatorFactory.get_available_evaluator_types()
        
        print(f"✅ Factory Pattern: Data types: {data_types}")
        print(f"✅ Factory Pattern: Evaluator types: {evaluator_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BERT Sentiment Classifier - Setup Test (Object-Oriented)")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_project_imports,
        test_object_oriented_classes,
        test_directories,
        test_files,
        test_cuda,
        test_architecture
    ]
    
    all_passed = True
    
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nObject-Oriented Architecture Features:")
        print("✅ Abstract base classes for extensibility")
        print("✅ Factory patterns for object creation")
        print("✅ Pipeline pattern for workflows")
        print("✅ Manager classes for resource management")
        print("✅ Separation of concerns")
        print("✅ Backward compatibility maintained")
        
        print("\nNext steps:")
        print("1. Train the model: python src/train.py")
        print("2. Evaluate the model: python src/evaluate.py")
        print("3. Make predictions: python src/predict.py")
        print("4. Launch the UI: python src/ui.py")
        
        print("\nAdvanced usage:")
        print("- Use ModelManager for multiple models")
        print("- Use DataManagerFactory for different datasets")
        print("- Use EvaluatorFactory for different evaluation types")
        print("- Extend base classes for custom implementations")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main() 