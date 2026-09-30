import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent   # -> backend/

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "model.keras"
META_PATH = SAVED_MODELS_DIR / "meta.json"

with open(META_PATH, "r") as f:
    META = json.load(f)

INPUT_SIZE = tuple(META["input_size"])      # (32, 32)
LABELS = META["labels"]                     # {"0": "FAKE", "1": "REAL"}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024          # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

CORS_ORIGINS = [
    "http://localhost:5173",                # Vite dev server (React)
]