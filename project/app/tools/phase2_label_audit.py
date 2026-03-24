from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
LABEL_MAPPING_PATH = CONFIG_DIR / "phase2_label_mapping.yaml"
OUTPUT_PATH = DATA_DIR / "phase2_label_audit_report.json"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def suggest_mapping(label: str) -> dict[str, Any]:
    text = label.lower()

    if any(k in text for k in ["cat", "dog", "animal", "meow", "bleat", "pet"]):
        return {"category": "animal_presence", "weight": 1.0, "reason": "animal keyword"}
    if any(k in text for k in ["speech", "conversation", "talk", "voice"]):
        return {"category": "human_presence", "weight": 1.0, "reason": "human voice keyword"}
    if "silence" in text or "quiet" in text:
        return {"category": "quiet", "weight": 1.0, "reason": "quiet keyword"}
    if any(k in text for k in ["music", "instrument", "song"]):
        return {"category": "artificial_sound", "weight": 1.0, "reason": "music keyword"}
    if any(k in text for k in ["vehicle", "car", "engine", "train"]):
        return {"category": "urban_noise", "weight": 1.0, "reason": "vehicle keyword"}
    if any(k in text for k in ["inside", "room", "indoor"]):
        return {"category": "indoor_hint", "weight": 1.0, "reason": "indoor keyword"}
    if any(k in text for k in ["water", "river", "stream", "ocean", "sea"]):
        return {"category": "water_presence", "weight": 1.0, "reason": "water keyword"}
    if any(k in text for k in ["wind", "storm", "breeze"]):
        return {"category": "wind_presence", "weight": 1.0, "reason": "wind keyword"}
    if any(k in text for k in ["bird", "owl", "caw"]):
        return {"category": "bird_presence", "weight": 1.0, "reason": "bird keyword"}
    if any(k in text for k in ["insect", "cricket"]):
        return {"category": "insect_presence", "weight": 1.0, "reason": "insect keyword"}

    return {"category": "TODO", "weight": 1.0, "reason": "no rule matched"}


def main() -> None:
    mapping = load_yaml(LABEL_MAPPING_PATH)
    registered = set((mapping.get("labels") or {}).keys())

    label_counter: Counter[str] = Counter()
    label_audio_ids: dict[str, set[str]] = {}

    for path in sorted(ANALYSIS_DIR.glob("*.json")):
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[WARN] skip invalid json: {path} ({e})")
            continue

        audio_id = str(data.get("audio_id", path.stem))
        for seg in data.get("segments", []):
            for event in seg.get("top_events", []):
                label = event.get("label")
                if not label:
                    continue
                label = str(label)
                label_counter[label] += 1
                label_audio_ids.setdefault(label, set()).add(audio_id)

    unknown_labels = []
    for label, count in label_counter.most_common():
        if label in registered:
            continue
        unknown_labels.append({
            "label": label,
            "count": count,
            "audio_ids": sorted(label_audio_ids.get(label, set())),
            "suggestion": suggest_mapping(label),
        })

    report = {
        "analysis_file_count": len(list(ANALYSIS_DIR.glob("*.json"))),
        "registered_label_count": len(registered),
        "observed_label_count": len(label_counter),
        "unknown_label_count": len(unknown_labels),
        "unknown_labels": unknown_labels,
    }

    write_json(OUTPUT_PATH, report)

    print(json.dumps({
        "output_path": str(OUTPUT_PATH),
        "unknown_label_count": len(unknown_labels),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
