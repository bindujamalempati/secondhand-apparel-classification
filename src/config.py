from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "sample_images"

TARGET_CLASSES = ["Tshirts", "Shirts", "Dresses", "Jeans", "Jackets"]
IMAGE_SIZE = (96, 96)
RANDOM_STATE = 42
