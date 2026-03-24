import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# =========================
# 設定
# =========================
DRY_DIR = Path("data/dry")
OUT_ROOT = Path("data")
LABELS_CSV = OUT_ROOT / "labels.csv"
META_JSONL = OUT_ROOT / "spatial_meta.jsonl"

TARGET_SR = 44100
SCENE_SECONDS = 30.0
SCENE_SAMPLES = int(TARGET_SR * SCENE_SECONDS)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SCENES_PER_SPACE = 60   # 5空間 × 60 = 300シーン
MIN_EVENTS = 3
MAX_EVENTS = 7

SPACES = [
    "dense_forest",
    "open_field",
    "mountain_slope",
    "river_side",
    "cave",
]

# 空間ごとの音響パラメータ
SPACE_PROFILES = {
    "dense_forest": {
        "rt60": 0.35,
        "early_reflections": 4,
        "reflection_gain": 0.20,
        "noise_level": 0.004,
        "distance_range": (2.0, 12.0),
    },
    "open_field": {
        "rt60": 0.12,
        "early_reflections": 1,
        "reflection_gain": 0.05,
        "noise_level": 0.002,
        "distance_range": (3.0, 20.0),
    },
    "mountain_slope": {
        "rt60": 0.45,
        "early_reflections": 3,
        "reflection_gain": 0.18,
        "noise_level": 0.003,
        "distance_range": (4.0, 18.0),
    },
    "river_side": {
        "rt60": 0.28,
        "early_reflections": 2,
        "reflection_gain": 0.10,
        "noise_level": 0.006,
        "distance_range": (2.0, 15.0),
    },
    "cave": {
        "rt60": 1.80,
        "early_reflections": 8,
        "reflection_gain": 0.35,
        "noise_level": 0.002,
        "distance_range": (1.5, 10.0),
    },
}


# =========================
# 基本関数
# =========================
def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def normalize_peak(x: np.ndarray, peak=0.95) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-9
    return (x / m) * peak


def choose_split() -> str:
    r = random.random()
    if r < TRAIN_RATIO:
        return "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"


# =========================
# 疑似RIR生成
# =========================
def make_synthetic_stereo_rir(
    sr: int,
    rt60: float,
    early_reflections: int,
    reflection_gain: float,
    pan: float,
):
    """
    pan: -1.0 (left) ～ +1.0 (right)
    """
    rir_len = int(sr * min(max(rt60 * 1.5, 0.3), 3.0))
    t = np.arange(rir_len) / sr

    # 減衰エンベロープ
    decay = np.exp(-6.91 * t / max(rt60, 1e-3))

    rir_l = np.zeros(rir_len, dtype=np.float32)
    rir_r = np.zeros(rir_len, dtype=np.float32)

    # 直接音
    direct_l = 1.0 * (1.0 - max(0.0, pan) * 0.35)
    direct_r = 1.0 * (1.0 + min(0.0, pan) * -0.35)
    rir_l[0] = direct_l
    rir_r[0] = direct_r

    # 初期反射
    for i in range(early_reflections):
        delay_ms = random.uniform(12, 90) * (i + 1) / early_reflections
        idx = min(int(sr * delay_ms / 1000.0), rir_len - 1)
        gain = reflection_gain * random.uniform(0.5, 1.0)

        # 少し左右差をつける
        l_gain = gain * (1.0 - max(0.0, pan) * 0.4)
        r_gain = gain * (1.0 + min(0.0, pan) * -0.4)
        rir_l[idx] += l_gain
        rir_r[idx] += r_gain

    # 残響尾を薄く追加
    tail_noise_l = np.random.randn(rir_len).astype(np.float32) * 0.015
    tail_noise_r = np.random.randn(rir_len).astype(np.float32) * 0.015
    rir_l += tail_noise_l * decay
    rir_r += tail_noise_r * decay

    # 軽く正規化
    rir_l /= (np.max(np.abs(rir_l)) + 1e-9)
    rir_r /= (np.max(np.abs(rir_r)) + 1e-9)

    return np.stack([rir_l, rir_r], axis=1)


def convolve_stereo(audio_st: np.ndarray, rir_st: np.ndarray) -> np.ndarray:
    y_l = fftconvolve(audio_st[:, 0], rir_st[:, 0], mode="full")
    y_r = fftconvolve(audio_st[:, 1], rir_st[:, 1], mode="full")
    return np.stack([y_l, y_r], axis=1)


def apply_distance(audio_st: np.ndarray, distance_m: float) -> np.ndarray:
    d = max(distance_m, 1.0)
    gain = 1.0 / (d ** 1.1)
    return audio_st * gain


def apply_pan(audio_st: np.ndarray, pan: float) -> np.ndarray:
    # equal power pan
    left = math.sqrt((1 - pan) / 2)
    right = math.sqrt((1 + pan) / 2)
    out = np.zeros_like(audio_st)
    mono = audio_st.mean(axis=1)
    out[:, 0] = mono * left
    out[:, 1] = mono * right
    return out


def add_background_noise(scene: np.ndarray, level: float) -> np.ndarray:
    noise = np.random.randn(*scene.shape).astype(np.float32) * level
    return scene + noise


# =========================
# シーン生成
# =========================
def build_scene(space_label: str, dry_files: list[Path]) -> tuple[np.ndarray, dict]:
    profile = SPACE_PROFILES[space_label]
    scene = np.zeros((SCENE_SAMPLES, 2), dtype=np.float32)

    num_events = random.randint(MIN_EVENTS, MAX_EVENTS)
    event_metas = []

    chosen = random.sample(dry_files, k=min(num_events, len(dry_files)))

    for dry_path in chosen:
        x, sr = sf.read(dry_path)
        if sr != TARGET_SR:
            raise ValueError(f"sample rate mismatch: {dry_path} sr={sr}, expected={TARGET_SR}")

        x = to_stereo(x).astype(np.float32)

        # 配置位置
        start_sec = random.uniform(0.0, max(0.1, SCENE_SECONDS - 8.0))
        start_idx = int(start_sec * TARGET_SR)

        # 空間パラメータ
        distance_m = random.uniform(*profile["distance_range"])
        pan = random.uniform(-0.9, 0.9)

        # パン → RIR → 距離
        x_pan = apply_pan(x, pan)
        rir = make_synthetic_stereo_rir(
            sr=TARGET_SR,
            rt60=profile["rt60"],
            early_reflections=profile["early_reflections"],
            reflection_gain=profile["reflection_gain"],
            pan=pan,
        )
        y = convolve_stereo(x_pan, rir)
        y = apply_distance(y, distance_m)

        end_idx = min(start_idx + len(y), SCENE_SAMPLES)
        valid_len = end_idx - start_idx
        if valid_len > 0:
            scene[start_idx:end_idx] += y[:valid_len]

        event_metas.append({
            "dry_file": str(dry_path),
            "start_sec": round(start_sec, 3),
            "distance_m": round(distance_m, 3),
            "pan": round(pan, 3),
        })

    scene = add_background_noise(scene, profile["noise_level"])
    scene = normalize_peak(scene, 0.95)

    meta = {
        "space_label": space_label,
        "scene_seconds": SCENE_SECONDS,
        "num_events": len(event_metas),
        "events": event_metas,
        "profile": profile,
    }
    return scene, meta


def main():
    random.seed(42)
    np.random.seed(42)
    ensure_dirs()

    dry_files = sorted(DRY_DIR.glob("*.wav"))
    if not dry_files:
        raise FileNotFoundError(f"No dry wav files found in {DRY_DIR}")

    label_rows = []
    scene_index = 0
    started = time.time()

    with open(META_JSONL, "w", encoding="utf-8") as meta_f:
        for space_label in SPACES:
            print(f"[INFO] generating space={space_label}")

            for _ in range(SCENES_PER_SPACE):
                scene, meta = build_scene(space_label, dry_files)
                split = choose_split()

                rel_path = f"{split}/scene_{scene_index:05d}.wav"
                out_path = OUT_ROOT / rel_path
                sf.write(out_path, scene, TARGET_SR)

                label_rows.append([rel_path, space_label])

                meta_record = {
                    "file": rel_path,
                    "label": space_label,
                    "created_at_unix": int(time.time()),
                    **meta,
                }
                meta_f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

                scene_index += 1

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerows(label_rows)

    elapsed = round(time.time() - started, 2)
    print(f"[DONE] scenes={scene_index}, labels={LABELS_CSV}, meta={META_JSONL}, elapsed={elapsed}s")


if __name__ == "__main__":
    main()