# BERT Sentiment Classifier

A fully open-source sentiment analysis project that uses a pretrained Hugging Face transformer model (BERT) to classify text into positive or negative sentiment. Built with modern object-oriented design principles for extensibility and maintainability.

## 🎯 Project Overview

This project implements a binary sentiment classifier using the BERT (Bidirectional Encoder Representations from Transformers) model. It provides a complete pipeline for training, evaluation, and inference, along with a user-friendly web interface.

### Features
- **Model**: Uses `bert-base-uncased` from Hugging Face Transformers
- **Dataset**: IMDB sentiment dataset for training and evaluation
- **Framework**: PyTorch with Hugging Face Trainer
- **UI**: Gradio web interface for easy interaction
- **Evaluation**: Comprehensive metrics including accuracy, F1-score, and confusion matrix
- **Security**: Environment variable management with no exposed secrets
- **Architecture**: Object-oriented design with abstract base classes, factory patterns, and pipeline workflows

### Object-Oriented Architecture
- **Abstract Base Classes**: Extensible design for models, data managers, and evaluators
- **Factory Patterns**: Easy creation of different model and data manager types
- **Pipeline Pattern**: Clean workflow management for training, evaluation, and prediction
- **Manager Classes**: Resource management for models and datasets
- **Separation of Concerns**: Clear boundaries between data, models, evaluation, and UI

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pathupally/bert-sentiment-classifier

cd bert-sentiment-classifier
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env file if you need to use private models
```

5. Test the setup:
```bash
python3 test_setup.py
```

## 📊 How to Train

Train the model on the IMDB dataset using the object-oriented pipeline:

```bash
python3 src/train.py
```

Or with custom parameters:
```bash
python3 src/train.py --data-type imdb --model-name my_bert_model
```

The training pipeline will:
- Download and preprocess the IMDB dataset
- Fine-tune the BERT model using the ModelTrainer class
- Save checkpoints during training
- Save the final model in `models/`
- Evaluate the model automatically

### Training Configuration

You can modify training parameters in `src/config.py`:
- Model name
- Batch size
- Learning rate
- Number of epochs
- Max sequence length

## 📈 How to Evaluate

Evaluate the trained model using the evaluation pipeline:

```bash
python3 src/evaluate.py
```

Or with custom model path:
```bash
python3 src/evaluate.py --model-path models/my_model --data-type imdb
```

This will generate:
- Accuracy, precision, recall, and F1-score
- ROC-AUC curve and confusion matrix visualization
- Detailed evaluation report
- Confidence distribution analysis

## 🔮 How to Predict

### Command Line Interface

Predict sentiment for a single text:

```bash
python3 src/predict.py "This movie was absolutely fantastic!"
```

### Programmatic Usage

```python
from src.predict import SentimentPredictionPipeline

# Create prediction pipeline
pipeline = SentimentPredictionPipeline()
pipeline.load_model()

# Single prediction
result = pipeline.predict_single("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch prediction
results = pipeline.predict_batch([
    "This is great!",
    "This is terrible.",
    "This is okay."
])
```

### Advanced Usage with Model Manager

```python
from src.models import ModelManager

# Create model manager
manager = ModelManager()

# Load multiple models
manager.load_model("bert_model", "models/bert_sentiment")
manager.load_model("custom_model", "models/custom_sentiment")

# Switch between models
manager.set_current_model("bert_model")
result1 = manager.predict_with_current("Great movie!")

manager.set_current_model("custom_model")
result2 = manager.predict_with_current("Great movie!")
```

## 🌐 How to Launch UI

Start the Gradio web interface:

```bash
python src/ui.py
```

Then open your browser to the provided URL (usually `http://127.0.0.1:7860`).

The UI provides:
- Text input for sentiment analysis
- Real-time prediction with confidence scores
- Batch processing capability
- Model information display
- Object-oriented architecture showcase

## 📁 Project Structure

```
bert-sentiment-classifier/
├── data/                    # Dataset storage
├── models/                  # Trained model weights
├── outputs/                 # Evaluation results and plots
├── src/
│   ├── config.py           # Configuration settings
│   ├── utils.py            # Helper functions
│   ├── models.py           # Model classes and managers
│   ├── data_manager.py     # Data management classes
│   ├── evaluator.py        # Evaluation classes
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation pipeline
│   ├── predict.py          # Prediction pipeline
│   └── ui.py               # Gradio web interface
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── test_setup.py          # Setup verification script
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🏗️ Architecture Overview

### Core Classes

#### Models (`src/models.py`)
- `BaseSentimentModel`: Abstract base class for sentiment models
- `BERTSentimentModel`: Concrete BERT implementation
- `ModelManager`: Manages multiple model instances
- `ModelTrainer`: Handles model training workflows

#### Data Management (`src/data_manager.py`)
- `BaseDataManager`: Abstract base class for data management
- `IMDBDataManager`: IMDB dataset implementation
- `CustomDataManager`: Support for custom datasets
- `DataManagerFactory`: Factory for creating data managers

#### Evaluation (`src/evaluator.py`)
- `BaseEvaluator`: Abstract base class for evaluation
- `SentimentEvaluator`: Sentiment-specific evaluation
- `EvaluatorFactory`: Factory for creating evaluators

#### Pipelines
- `SentimentTrainingPipeline`: Complete training workflow
- `SentimentEvaluationPipeline`: Complete evaluation workflow
- `SentimentPredictionPipeline`: Complete prediction workflow

### Design Patterns Used

1. **Abstract Base Classes**: For extensibility
2. **Factory Pattern**: For object creation
3. **Pipeline Pattern**: For workflow management
4. **Manager Pattern**: For resource management
5. **Strategy Pattern**: For different evaluation methods

## 📊 Sample Predictions

| Text | Sentiment | Confidence |
|------|-----------|------------|
| "This movie was absolutely fantastic!" | Positive | 0.98 |
| "I hated every minute of this film." | Negative | 0.95 |
| "The acting was okay but the plot was weak." | Negative | 0.72 |
| "A masterpiece of modern cinema." | Positive | 0.89 |

## 🔧 Configuration

Key configuration options in `src/config.py`:

- `MODEL_NAME`: Hugging Face model identifier
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Learning rate for optimization
- `NUM_EPOCHS`: Number of training epochs

## 🛡️ Security

This project follows security best practices:
- No API keys or secrets in the code
- Environment variables for sensitive data
- `.gitignore` prevents accidental commits of secrets
- All dependencies are open-source

## 📈 Performance

Typical performance on IMDB test set:
- **Accuracy**: ~92-94%
- **F1-Score**: ~92-94%
- **Training Time**: ~30-60 minutes (depending on hardware)

## 🔄 Extending the Project

### Adding New Models

```python
from src.models import BaseSentimentModel

class MyCustomModel(BaseSentimentModel):
    def load_model(self):
        # Custom model loading logic
        pass
    
    def predict(self, text):
        # Custom prediction logic
        pass
```

### Adding New Datasets

```python
from src.data_manager import BaseDataManager

class MyDatasetManager(BaseDataManager):
    def load_data(self):
        # Custom data loading logic
        pass
    
    def split_data(self, dataset):
        # Custom splitting logic
        pass
```

### Adding New Evaluators

```python
from src.evaluator import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, test_dataset):
        # Custom evaluation logic
        pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the object-oriented design principles
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Google Research for BERT
- IMDB dataset creators
- Gradio for the web interface
- The open-source community for design patterns and best practices

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

## 🚀 Advanced Features

### Multiple Model Support
```python
# Load and manage multiple models
manager = ModelManager()
manager.load_model("bert_base", "models/bert_base")
manager.load_model("bert_large", "models/bert_large")

# Compare predictions
result1 = manager.get_model("bert_base").predict("Great movie!")
result2 = manager.get_model("bert_large").predict("Great movie!")
```

### Custom Dataset Support
```python
# Use custom dataset
data_manager = DataManagerFactory.create_data_manager("custom")
data_manager.load_from_csv("my_data.csv", text_column="review", label_column="sentiment")
splits = data_manager.split_data()
```

### Custom Evaluation Metrics
```python
# Create custom evaluator
evaluator = EvaluatorFactory.create_evaluator("sentiment", model=model)
results = evaluator.evaluate(test_dataset)
# Access comprehensive metrics and visualizations
``` 
