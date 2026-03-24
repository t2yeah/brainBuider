from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
INTERPRET_DIR = DATA_DIR / "interpret"
STORY_DIR = DATA_DIR / "story"

SCENE_TEMPLATE_PATH = CONFIG_DIR / "phase3_scene_templates.yaml"


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


def resolve_interpreted_path(audio_id: str) -> Path:
    path = INTERPRET_DIR / audio_id / "interpreted.json"
    if not path.exists():
        raise FileNotFoundError(f"interpreted.json not found: {path}")
    return path


def get_scene_template(scene_id: str, scene_templates: dict[str, Any]) -> dict[str, Any]:
    templates = scene_templates.get("scene_templates", {})
    return templates.get(scene_id) or templates.get("unknown", {})


def build_segment_story(segment: dict[str, Any], scene_templates: dict[str, Any]) -> dict[str, Any]:
    scene_id = segment.get("scene_id", "unknown")
    template = get_scene_template(scene_id, scene_templates)

    return {
        "segment_id": segment["segment_id"],
        "segment_index": segment.get("segment_index"),
        "scene_id": scene_id,
        "display_name": template.get("display_name", scene_id),
        "mood": template.get("mood", ""),
        "story_text": template.get("segment_template", segment.get("description", "")),
        "categories": segment.get("categories", {}),
        "source_description": segment.get("description", ""),
    }


def unique_keywords(keyword_lists: list[list[str]]) -> list[str]:
    seen = set()
    result = []
    for keywords in keyword_lists:
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
    return result


def build_story(audio_id: str) -> dict[str, Any]:
    interpreted_path = resolve_interpreted_path(audio_id)
    interpreted = load_json(interpreted_path)
    scene_templates = load_yaml(SCENE_TEMPLATE_PATH)

    overall_scene = interpreted.get("overall_scene", {})
    overall_scene_id = overall_scene.get("scene_id", "unknown")
    overall_template = get_scene_template(overall_scene_id, scene_templates)

    segment_stories = [
        build_segment_story(seg, scene_templates)
        for seg in interpreted.get("segments", [])
    ]

    keyword_lists = []
    for seg in interpreted.get("segments", []):
        scene_id = seg.get("scene_id", "unknown")
        template = get_scene_template(scene_id, scene_templates)
        keyword_lists.append(template.get("manga_prompt_keywords", []))

    overall_keywords = overall_template.get("manga_prompt_keywords", [])
    manga_keywords = unique_keywords([overall_keywords] + keyword_lists)

    story = {
        "audio_id": audio_id,
        "source_interpreted_path": str(interpreted_path),
        "overall_scene_id": overall_scene_id,
        "overall_display_name": overall_template.get("display_name", overall_scene_id),
        "overall_mood": overall_template.get("mood", ""),
        "overall_story": overall_template.get("overall_template", overall_scene.get("summary", "")),
        "overall_summary_from_phase2": overall_scene.get("summary", ""),
        "segment_count": len(segment_stories),
        "segments": segment_stories,
        "manga_prompt_base": ", ".join(manga_keywords),
    }

    return story


def output_path(audio_id: str) -> Path:
    return STORY_DIR / audio_id / "story.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase3 story from interpreted result")
    parser.add_argument("--audio-id", required=True, help="audio_id")
    args = parser.parse_args()

    story = build_story(args.audio_id)
    out = output_path(args.audio_id)
    write_json(out, story)

    print(json.dumps({
        "audio_id": args.audio_id,
        "story_path": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
