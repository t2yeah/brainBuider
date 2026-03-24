from pydantic import BaseModel

class UploadResponse(BaseModel):
    audio_id: str
    original_filename: str
    saved_path: str
    content_type: str | None
    size_bytes: int
    status: str
