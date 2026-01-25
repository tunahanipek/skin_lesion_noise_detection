"""Configuration settings for Skin Quality QA"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Model Configuration
MODEL_NAME = "gemini-2.0-flash"
IMAGE_ARTIFACTS = ["blurry", "hairy", "bubble", "ruler", "vignette", "gel_border"]

# Data paths
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
METADATA_ENRICHED_CSV = os.path.join(DATA_DIR, "metadata_enriched.csv")

# Model paths
MODELS_DIR = "./models"

# Processing settings
DEFAULT_DAILY_LIMIT = 40  # Images per day (free tier safe limit)
REQUEST_DELAY = 15  # Seconds between requests (for RPM limit - increased for free tier)
RETRY_ATTEMPTS = 5
RETRY_DELAY_BASE = 60  # Base delay in seconds for exponential backoff

# Log settings
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
