from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
NORMALIZED_DIR = DATA_DIR / "normalized"
SEGMENTS_DIR = DATA_DIR / "segments"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024
TARGET_SAMPLE_RATE = 32000
TARGET_CHANNELS = 1
SEGMENT_SECONDS = 5
