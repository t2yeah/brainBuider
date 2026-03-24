from pathlib import Path
import math
import subprocess
import soundfile as sf

from app.core.config import (
    UPLOAD_DIR,
    NORMALIZED_DIR,
    SEGMENTS_DIR,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    SEGMENT_SECONDS,
)

ALLOWED_INPUT_EXTS = [".wav", ".mp3", ".m4a", ".webm"]


def find_uploaded_file(audio_id: str) -> Path:
    for ext in ALLOWED_INPUT_EXTS:
        p = UPLOAD_DIR / f"{audio_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"uploaded file not found for audio_id={audio_id}")


def normalize_audio(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-vn",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_audio_info(wav_path: Path) -> tuple[float, int, int]:
    data, sr = sf.read(str(wav_path))
    frames = len(data)
    duration = frames / sr

    if len(data.shape) == 1:
        channels = 1
    else:
        channels = data.shape[1]

    return duration, sr, channels


def split_audio_segments(input_wav: Path, audio_id: str) -> list[str]:
    seg_dir = SEGMENTS_DIR / audio_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    duration, _, _ = get_audio_info(input_wav)
    segment_count = math.ceil(duration / SEGMENT_SECONDS)

    outputs = []

    for i in range(segment_count):
        start = i * SEGMENT_SECONDS
        out_path = seg_dir / f"segment_{i:03d}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ss",
            str(start),
            "-t",
            str(SEGMENT_SECONDS),
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs.append(str(out_path))

    return outputs


def preprocess_audio(audio_id: str) -> dict:
    input_path = find_uploaded_file(audio_id)
    normalized_path = NORMALIZED_DIR / f"{audio_id}.wav"

    normalize_audio(input_path, normalized_path)
    duration_sec, sample_rate, channels = get_audio_info(normalized_path)
    segments = split_audio_segments(normalized_path, audio_id)

    return {
        "audio_id": audio_id,
        "normalized_path": str(normalized_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": round(duration_sec, 3),
        "segment_seconds": SEGMENT_SECONDS,
        "segment_count": len(segments),
        "segments": segments,
        "status": "preprocessed",
    }


def run(audio_id: str) -> dict:
    """
    upload.py や他のサービスから呼ぶための統一エントリポイント
    """
    return preprocess_audio(audio_id)