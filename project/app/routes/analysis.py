from pathlib import Path
import json

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/analysis", tags=["analysis"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"


def load_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path.name}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"invalid json: {path.name}")


@router.get("/_health")
def health():
    return {"status": "ok"}


@router.get("/{audio_id}/status")
def get_analysis_status(audio_id: str):
    status_path = RESULT_ROOT / audio_id / "pipeline_status.json"
    return load_json(status_path)


@router.get("/{audio_id}")
def get_analysis_result(audio_id: str):
    result_path = RESULT_ROOT / audio_id / "10_final_result.json"
    return load_json(result_path)