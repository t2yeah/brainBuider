import argparse
import glob
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa

from panns_inference import SoundEventDetection, labels

def load_audio_32k_mono(path: str) -> np.ndarray:
    # PANNs expects 32 kHz mono
    y, _ = librosa.load(path, sr=32000, mono=True)
    # shape: (1, samples)
    return y[None, :]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory containing wav files")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--topk", type=int, default=10, help="Top-K labels to output per file")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern (default: *.wav)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer_panns] device={device}")

    # Init model
    sed = SoundEventDetection(checkpoint_path=None, device=device)

    # Collect inputs
    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files found: {args.in_dir}/{args.pattern}")

    rows = []
    for f in files:
        audio = load_audio_32k_mono(f)

        # Inference
        framewise_output, clipwise_output = sed.inference(audio)
        clip = clipwise_output[0]  # (num_labels,)

        top_idx = np.argsort(-clip)[: args.topk]
        top_labels = [labels[i] for i in top_idx]
        top_scores = [float(clip[i]) for i in top_idx]

        rows.append({
            "file": Path(f).name,
            "top_labels": json.dumps(top_labels, ensure_ascii=False),
            "top_scores": json.dumps(top_scores, ensure_ascii=False),
        })

        print(f"[OK] {Path(f).name}  top1={top_labels[0]} ({top_scores[0]:.3f})")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
