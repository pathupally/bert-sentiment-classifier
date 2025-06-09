"""
Configuration settings for the BERT Sentiment Classifier project.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))

# Training Configuration
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "500"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "500"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "500"))

# Evaluation Configuration
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))

# Dataset Configuration
DATASET_NAME = "imdb"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_classifier")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer")

# UI Configuration
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Hugging Face Configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# Sentiment labels
SENTIMENT_LABELS = ["negative", "positive"]
NUM_LABELS = len(SENTIMENT_LABELS)

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"

# Training arguments for Hugging Face Trainer
TRAINING_ARGS = {
    "output_dir": MODEL_PATH,
    "num_train_epochs": NUM_EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "weight_decay": WEIGHT_DECAY,
    "logging_dir": os.path.join(OUTPUTS_DIR, "logs"),
    "logging_steps": 100,
    "save_steps": SAVE_STEPS,
    "eval_steps": EVAL_STEPS,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "fp16": True,  # Use mixed precision if available
    "report_to": None,  # Disable wandb/tensorboard reporting
} 