from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
INTERPRET_DIR = DATA_DIR / "interpret"
EXPECTED_JSONL_PATH = DATA_DIR / "phase2_expected_interpretations.jsonl"
DRAFT_JSONL_PATH = DATA_DIR / "phase2_expected_interpretations_draft.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_existing_audio_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    audio_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            audio_id = row.get("audio_id")
            if audio_id:
                audio_ids.add(str(audio_id))
    return audio_ids


def build_expected_row(interpreted: dict[str, Any]) -> dict[str, Any]:
    audio_id = str(interpreted["audio_id"])
    overall_scene = interpreted.get("overall_scene", {}).get("scene_id", "unknown")

    expected_segments = []
    for seg in interpreted.get("segments", []):
        expected_segments.append({
            "segment_id": seg["segment_id"],
            "scene_id": seg["scene_id"],
        })

    return {
        "audio_id": audio_id,
        "status": "draft",
        "expected_overall_scene": overall_scene,
        "expected_segments": expected_segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expected draft jsonl from interpreted outputs")
    parser.add_argument("--force", action="store_true", help="include already registered audio_ids")
    args = parser.parse_args()

    existing_audio_ids = load_existing_audio_ids(EXPECTED_JSONL_PATH)

    rows = []
    for interpreted_path in sorted(INTERPRET_DIR.glob("*/interpreted.json")):
        try:
            interpreted = load_json(interpreted_path)
        except Exception as e:
            print(f"[WARN] skip invalid interpreted file: {interpreted_path} ({e})")
            continue

        audio_id = str(interpreted.get("audio_id", ""))
        if not audio_id:
            continue

        if (not args.force) and (audio_id in existing_audio_ids):
            continue

        rows.append(build_expected_row(interpreted))

    DRAFT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DRAFT_JSONL_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output_path": str(DRAFT_JSONL_PATH),
        "draft_count": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
