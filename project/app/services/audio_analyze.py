from __future__ import annotations

from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import torch
import librosa

from panns_inference import AudioTagging, labels
from app.core.config import SEGMENTS_DIR


TOP_K = 5
GLOBAL_TOP_K = 10

PANN_SAMPLE_RATE = 32000
PANN_INPUT_SECONDS = 10
PANN_INPUT_SAMPLES = PANN_SAMPLE_RATE * PANN_INPUT_SECONDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# まずは安定優先でCPU
# 必要なら後で cuda に変更
_DEVICE = "cpu"
_MODEL = AudioTagging(checkpoint_path=None, device=_DEVICE)


def _extract_segment_index(path: Path) -> int:
    m = re.search(r"segment_(\d+)\.wav$", path.name)
    return int(m.group(1)) if m else -1


def _list_segment_files(audio_id: str) -> list[Path]:
    seg_dir = SEGMENTS_DIR / audio_id
    if not seg_dir.exists():
        raise FileNotFoundError(f"segment directory not found for audio_id={audio_id}: {seg_dir}")

    files = sorted(seg_dir.glob("segment_*.wav"))
    if not files:
        raise FileNotFoundError(f"no segment wav files found for audio_id={audio_id}: {seg_dir}")

    return files


def _load_audio_for_panns(wav_path: Path) -> np.ndarray:
    """
    PANNs 用に:
    - mono
    - 32kHz
    - float32
    - 10秒に揃える（短ければゼロ埋め、長ければ先頭10秒）
    """
    audio, _ = librosa.load(str(wav_path), sr=PANN_SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)

    current_len = len(audio)
    if current_len < PANN_INPUT_SAMPLES:
        pad_width = PANN_INPUT_SAMPLES - current_len
        audio = np.pad(audio, (0, pad_width), mode="constant")
    elif current_len > PANN_INPUT_SAMPLES:
        audio = audio[:PANN_INPUT_SAMPLES]

    # (batch, samples)
    return audio[None, :]


def _analyze_one_file(wav_path: Path) -> list[dict]:
    audio = _load_audio_for_panns(wav_path)
    clipwise_output, _embedding = _MODEL.inference(audio)

    clipwise_output = np.asarray(clipwise_output)
    if clipwise_output.ndim != 2 or clipwise_output.shape[0] < 1:
        raise RuntimeError(f"unexpected clipwise_output shape: {clipwise_output.shape}")

    scores = clipwise_output[0]
    if len(scores) != len(labels):
        raise RuntimeError(
            f"label length mismatch: scores={len(scores)}, labels={len(labels)}"
        )

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    top_events = []
    for idx in top_indices:
        top_events.append({
            "label": str(labels[idx]),
            "score": round(float(scores[idx]), 4),
        })

    return top_events


def _build_global_top_events(segments: list[dict]) -> list[dict]:
    """
    全セグメントの top_events を集約して、
    後続で使いやすい global_top_events を作る
    """
    score_sum = defaultdict(float)
    hit_count = defaultdict(int)

    for seg in segments:
        for ev in seg.get("top_events", []):
            label = str(ev["label"])
            score = float(ev["score"])
            score_sum[label] += score
            hit_count[label] += 1

    merged = []
    for label, total_score in score_sum.items():
        count = hit_count[label]
        merged.append({
            "label": label,
            "score_sum": round(total_score, 4),
            "hit_count": count,
            "mean_score": round(total_score / count, 4),
        })

    merged.sort(key=lambda x: (x["hit_count"], x["score_sum"]), reverse=True)
    return merged[:GLOBAL_TOP_K]


def analyze_audio_segments(audio_id: str) -> dict:
    segment_files = _list_segment_files(audio_id)

    results = []
    for seg_path in segment_files:
        top_events = _analyze_one_file(seg_path)
        results.append({
            "segment_index": _extract_segment_index(seg_path),
            "segment_id": f"segment_{_extract_segment_index(seg_path):03d}",
            "segment_path": str(seg_path),
            "top_events": top_events,
        })

    return {
        "audio_id": audio_id,
        "analysis_version": "panns_v1",
        "model": "panns_inference.AudioTagging",
        "device": _DEVICE,
        "sample_rate": PANN_SAMPLE_RATE,
        "top_k_per_segment": TOP_K,
        "segment_count": len(results),
        "segments": results,
        "global_top_events": _build_global_top_events(results),
        "status": "analyzed",
    }


def _result_dir(audio_id: str) -> Path:
    path = RESULTS_DIR / audio_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_path(audio_id: str) -> Path:
    return _result_dir(audio_id) / "05_audio_events.json"


def save_result(audio_id: str, result: dict, pretty: bool = True) -> Path:
    out_path = _output_path(audio_id)
    with out_path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            json.dump(result, f, ensure_ascii=False)
    return out_path


def run(audio_id: str, pretty: bool = True) -> dict:
    result = analyze_audio_segments(audio_id)
    out_path = save_result(audio_id, result, pretty=pretty)
    print(f"[audio_analyze] saved -> {out_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True, help="audio_id")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = run(args.audio_id, pretty=args.pretty)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
