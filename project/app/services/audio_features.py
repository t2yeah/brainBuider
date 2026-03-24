import os
import json
import librosa
import numpy as np

SEGMENT_ROOT = "data/segments"
RESULT_ROOT = "data/results"


def compute_features(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    peak = np.max(np.abs(y))
    dynamic_range = np.max(rms) - np.min(rms)

    silence_threshold = 0.01
    silence_ratio = np.mean(rms < silence_threshold)

    return {
        "duration_sec": float(duration),
        "sample_rate": sr,

        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "peak_amplitude": float(peak),
        "dynamic_range": float(dynamic_range),

        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),

        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "spectral_flatness_mean": float(np.mean(flatness)),

        "onset_strength_mean": float(np.mean(onset)),
        "silence_ratio": float(silence_ratio)
    }


def run(audio_id: str):

    segment_dir = os.path.join(SEGMENT_ROOT, audio_id)
    result_dir = os.path.join(RESULT_ROOT, audio_id)

    os.makedirs(result_dir, exist_ok=True)

    segments = sorted(os.listdir(segment_dir))

    segment_features = []

    for seg in segments:

        if not seg.endswith(".wav"):
            continue

        seg_path = os.path.join(segment_dir, seg)

        features = compute_features(seg_path)

        segment_features.append({
            "segment_id": seg.replace(".wav", ""),
            "path": seg_path,
            "features": features
        })

    output = {
        "audio_id": audio_id,
        "segment_features": segment_features
    }

    save_path = os.path.join(result_dir, "04_features.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[audio_features] saved -> {save_path}")
