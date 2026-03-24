from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path.home() / "project"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
INTERPRET_DIR = DATA_DIR / "interpret"

LABEL_MAPPING_PATH = CONFIG_DIR / "phase2_label_mapping.yaml"
INTERPRET_RULES_PATH = CONFIG_DIR / "phase2_interpret_rules.yaml"
EXPECTED_JSONL_PATH = DATA_DIR / "phase2_expected_interpretations.jsonl"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be object: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_analysis_path(audio_id: str, analysis_dir: Path) -> Path:
    direct = analysis_dir / f"{audio_id}.json"
    if direct.exists():
        return direct

    for path in analysis_dir.glob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        if data.get("audio_id") == audio_id:
            return path

    raise FileNotFoundError(f"audio_id={audio_id} not found under {analysis_dir}")


def top_events_to_label_dict(top_events: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(top_events, list):
        return result

    for item in top_events:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        score = item.get("score")
        if label is None or score is None:
            continue
        result[str(label)] = float(score)

    return result


def normalize_label_scores(
    raw_labels: dict[str, float],
    label_mapping: dict[str, Any],
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    mapping = label_mapping.get("labels", {})

    for eng_label, prob in raw_labels.items():
        rule = mapping.get(eng_label)
        if not rule:
            continue

        category = rule.get("category")
        weight = float(rule.get("weight", 1.0))
        if category:
            scores[category] += float(prob) * weight

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def rule_matches_scene(categories: dict[str, float], rule: dict[str, Any]) -> tuple[bool, float]:
    all_conditions = rule.get("all", {}) or {}
    any_conditions = rule.get("any", {}) or {}

    for cat, threshold in all_conditions.items():
        if float(categories.get(cat, 0.0)) < float(threshold):
            return False, 0.0

    if any_conditions:
        any_ok = any(float(categories.get(cat, 0.0)) >= float(thr) for cat, thr in any_conditions.items())
        if not any_ok:
            return False, 0.0

    score = 0.0
    for cat in set(list(all_conditions.keys()) + list(any_conditions.keys())):
        score += float(categories.get(cat, 0.0))

    score += float(rule.get("priority", 0)) / 1000.0
    return True, score


def interpret_segment(segment: dict[str, Any], interpret_rules: dict[str, Any]) -> dict[str, Any]:
    categories = segment["categories"]
    best_scene = None
    best_description = None
    best_score = -1.0

    for rule in interpret_rules.get("segment_rules", []):
        matched, score = rule_matches_scene(categories, rule)
        if matched and score > best_score:
            best_scene = rule["scene_id"]
            best_description = rule.get("description", "")
            best_score = score

    if best_scene is None:
        default_rule = interpret_rules.get("default_segment_scene", {}) or {}
        best_scene = default_rule.get("scene_id", "unknown")
        best_description = default_rule.get("description", "No interpretation")

    return {
        "audio_id": segment["audio_id"],
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
        "segment_path": segment.get("segment_path"),
        "scene_id": best_scene,
        "description": best_description,
        "categories": categories,
    }


def build_overall_scene(interpreted_segments: list[dict[str, Any]], interpret_rules: dict[str, Any]) -> dict[str, Any]:
    if not interpreted_segments:
        default_rule = interpret_rules.get("default_overall_scene", {}) or {}
        return {
            "scene_id": default_rule.get("scene_id", "unknown"),
            "summary": default_rule.get("summary", "No overall interpretation"),
            "scene_ratio": {},
            "segment_count": 0,
        }

    counter = Counter(seg["scene_id"] for seg in interpreted_segments)
    total = len(interpreted_segments)
    ratios = {scene_id: count / total for scene_id, count in counter.items()}

    for rule in interpret_rules.get("overall_rules", []):
        ratio_rule = rule.get("min_ratio_by_scene")
        if ratio_rule:
            ok = True
            for scene_id, min_ratio in ratio_rule.items():
                if ratios.get(scene_id, 0.0) < float(min_ratio):
                    ok = False
                    break
            if ok:
                return {
                    "scene_id": rule["scene_id"],
                    "summary": rule.get("summary", ""),
                    "scene_ratio": ratios,
                    "segment_count": total,
                }

        min_distinct_scenes = rule.get("min_distinct_scenes")
        if min_distinct_scenes is not None and len(counter.keys()) >= int(min_distinct_scenes):
            return {
                "scene_id": rule["scene_id"],
                "summary": rule.get("summary", ""),
                "scene_ratio": ratios,
                "segment_count": total,
            }

    default_rule = interpret_rules.get("default_overall_scene", {}) or {}
    return {
        "scene_id": default_rule.get("scene_id", "unknown"),
        "summary": default_rule.get("summary", "No overall interpretation"),
        "scene_ratio": ratios,
        "segment_count": total,
    }


def load_expected_for_audio(audio_id: str, expected_jsonl_path: Path) -> dict[str, Any] | None:
    if not expected_jsonl_path.exists():
        return None

    with expected_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("audio_id") == audio_id:
                return row
    return None


def compare_with_expected(
    audio_id: str,
    overall_scene: dict[str, Any],
    interpreted_segments: list[dict[str, Any]],
    expected_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if expected_row is None:
        return {
            "audio_id": audio_id,
            "has_expected": False,
            "overall_match": None,
            "segment_accuracy": None,
            "segment_total": 0,
            "segment_matched": 0,
            "mismatches": [],
        }

    expected_overall_scene = expected_row.get("expected_overall_scene")
    overall_match = overall_scene["scene_id"] == expected_overall_scene

    expected_segments = {
        item["segment_id"]: item["scene_id"]
        for item in expected_row.get("expected_segments", [])
        if "segment_id" in item and "scene_id" in item
    }

    matched = 0
    total = 0
    mismatches = []

    for seg in interpreted_segments:
        seg_id = seg["segment_id"]
        if seg_id not in expected_segments:
            continue

        total += 1
        actual_scene = seg["scene_id"]
        expected_scene = expected_segments[seg_id]
        if actual_scene == expected_scene:
            matched += 1
        else:
            mismatches.append({
                "segment_id": seg_id,
                "expected_scene_id": expected_scene,
                "actual_scene_id": actual_scene,
            })

    accuracy = (matched / total) if total > 0 else None

    return {
        "audio_id": audio_id,
        "has_expected": True,
        "overall_match": overall_match,
        "expected_overall_scene": expected_overall_scene,
        "actual_overall_scene": overall_scene["scene_id"],
        "segment_accuracy": accuracy,
        "segment_total": total,
        "segment_matched": matched,
        "mismatches": mismatches,
    }


def normalize_analysis(analysis_data: dict[str, Any], label_mapping: dict[str, Any]) -> dict[str, Any]:
    audio_id = analysis_data.get("audio_id")
    if not audio_id:
        raise ValueError("audio_id is required in analysis data")

    normalized_segments = []

    for seg in analysis_data.get("segments", []):
        segment_index = int(seg.get("segment_index", 0))
        segment_id = f"{audio_id}_seg_{segment_index:04d}"
        raw_labels = top_events_to_label_dict(seg.get("top_events", []))

        normalized_segments.append({
            "audio_id": audio_id,
            "segment_id": segment_id,
            "segment_index": segment_index,
            "segment_path": seg.get("segment_path"),
            "raw_labels": raw_labels,
            "categories": normalize_label_scores(raw_labels, label_mapping),
        })

    return {
        "audio_id": audio_id,
        "segment_count": len(normalized_segments),
        "segments": normalized_segments,
    }


def interpret_analysis(normalized_data: dict[str, Any], interpret_rules: dict[str, Any]) -> dict[str, Any]:
    audio_id = normalized_data["audio_id"]

    interpreted_segments = [
        interpret_segment(seg, interpret_rules)
        for seg in normalized_data.get("segments", [])
    ]
    overall_scene = build_overall_scene(interpreted_segments, interpret_rules)

    return {
        "audio_id": audio_id,
        "segment_count": len(interpreted_segments),
        "overall_scene": overall_scene,
        "segments": interpreted_segments,
    }


def build_output_paths(audio_id: str, interpret_dir: Path) -> dict[str, Path]:
    base = interpret_dir / audio_id
    return {
        "base_dir": base,
        "normalized": base / "normalized.json",
        "interpreted": base / "interpreted.json",
        "evaluation": base / "evaluation.json",
    }


def run_phase2_pipeline(audio_id: str) -> dict[str, Any]:
    analysis_path = resolve_analysis_path(audio_id, ANALYSIS_DIR)
    analysis_data = load_json(analysis_path)

    actual_audio_id = analysis_data.get("audio_id")
    if not actual_audio_id:
        raise ValueError(f"audio_id missing in file: {analysis_path}")

    label_mapping = load_yaml(LABEL_MAPPING_PATH)
    interpret_rules = load_yaml(INTERPRET_RULES_PATH)

    normalized = normalize_analysis(analysis_data, label_mapping)
    interpreted = interpret_analysis(normalized, interpret_rules)

    expected_row = load_expected_for_audio(actual_audio_id, EXPECTED_JSONL_PATH)
    evaluation = compare_with_expected(
        audio_id=actual_audio_id,
        overall_scene=interpreted["overall_scene"],
        interpreted_segments=interpreted["segments"],
        expected_row=expected_row,
    )

    output_paths = build_output_paths(actual_audio_id, INTERPRET_DIR)
    write_json(output_paths["normalized"], normalized)
    write_json(output_paths["interpreted"], interpreted)
    write_json(output_paths["evaluation"], evaluation)

    return {
        "audio_id": actual_audio_id,
        "input_path": str(analysis_path),
        "output_dir": str(output_paths["base_dir"]),
        "normalized_path": str(output_paths["normalized"]),
        "interpreted_path": str(output_paths["interpreted"]),
        "evaluation_path": str(output_paths["evaluation"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 interpret pipeline")
    parser.add_argument("--audio-id", required=True, help="audio_id to interpret")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase2_pipeline(audio_id=args.audio_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
