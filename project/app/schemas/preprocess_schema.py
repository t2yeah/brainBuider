from pydantic import BaseModel

class PreprocessResponse(BaseModel):
    audio_id: str
    normalized_path: str
    sample_rate: int
    channels: int
    duration_sec: float
    segment_seconds: int
    segment_count: int
    segments: list[str]
    status: str
