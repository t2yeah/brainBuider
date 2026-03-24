from pydantic import BaseModel


class EventScore(BaseModel):
    label: str
    score: float


class SegmentAnalysis(BaseModel):
    segment_index: int
    segment_path: str
    top_events: list[EventScore]


class AnalyzeResponse(BaseModel):
    audio_id: str
    segment_count: int
    segments: list[SegmentAnalysis]
    status: str
