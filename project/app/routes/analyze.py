from fastapi import APIRouter, HTTPException

from app.schemas.analyze_schema import AnalyzeResponse
from app.services.audio_analyze import analyze_audio_segments

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze/{audio_id}", response_model=AnalyzeResponse)
def run_analyze(audio_id: str):
    try:
        result = analyze_audio_segments(audio_id)
        return AnalyzeResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze failed: {e}")
