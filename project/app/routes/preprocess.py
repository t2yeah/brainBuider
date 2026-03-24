from fastapi import APIRouter, HTTPException
from app.schemas.preprocess_schema import PreprocessResponse
from app.services.audio_preprocess import preprocess_audio

router = APIRouter(prefix="/api", tags=["preprocess"])


@router.post("/preprocess/{audio_id}", response_model=PreprocessResponse)
def run_preprocess(audio_id: str):
    try:
        result = preprocess_audio(audio_id)
        return PreprocessResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preprocess failed: {e}")
