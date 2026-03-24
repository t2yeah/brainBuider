from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

PANNS_CACHE_DIR = MODELS_DIR / "panns_cache"

ANIMAGINE_MODEL_DIR = MODELS_DIR / "animagine_xl_3_1"
ANIMAGINE_OUTPUT_DIR = OUTPUTS_DIR / "animagine"
ANIMAGINE_PROMPTS_DIR = ANIMAGINE_OUTPUT_DIR / "prompts"
ANIMAGINE_IMAGES_DIR = ANIMAGINE_OUTPUT_DIR / "images"
ANIMAGINE_METADATA_DIR = ANIMAGINE_OUTPUT_DIR / "metadata"
ANIMAGINE_DEBUG_DIR = ANIMAGINE_OUTPUT_DIR / "debug"

def ensure_animagine_dirs():
    ANIMAGINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANIMAGINE_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ANIMAGINE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANIMAGINE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ANIMAGINE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)