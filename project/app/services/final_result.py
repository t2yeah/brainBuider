from pathlib import Path
import json
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_title(scene: Dict[str, Any], manga_prompt: Dict[str, Any]) -> str:
    """
    タイトルの優先順位:
    1. 08_manga_prompt.json の title
    2. 08_manga_prompt.json の manga_title
    3. 07_scene_interpretation.json の scene_title
    4. デフォルト
    """
    return (
        manga_prompt.get("title")
        or manga_prompt.get("manga_title")
        or scene.get("scene_title")
        or "無題"
    )


def run(audio_id: str) -> None:
    result_dir = RESULT_ROOT / audio_id

    features_path = result_dir / "04_features.json"
    similarity_path = result_dir / "05_space_similarity.json"
    judgement_path = result_dir / "06_space_judgement.json"
    scene_path = result_dir / "07_scene_interpretation.json"
    onomatopoeia_path = result_dir / "08_onomatopoeia.json"
    manga_prompt_path = result_dir / "08_manga_prompt.json"
    output_path = result_dir / "10_final_result.json"

    features = load_json(features_path)
    similarity = load_json(similarity_path)
    judgement = load_json(judgement_path)
    scene = load_json(scene_path)
    onomatopoeia = load_json(onomatopoeia_path)
    manga_prompt = load_json(manga_prompt_path)

    segment_features = features.get("segment_features", [])
    segment_count = len(segment_features)

    if segment_features:
        first_features = segment_features[0].get("features", {})
        sample_rate = first_features.get("sample_rate")
    else:
        sample_rate = None

    total_duration = 0.0
    for item in segment_features:
        total_duration += float(item.get("features", {}).get("duration_sec", 0.0))

    final_title = pick_title(scene, manga_prompt)

    final_output = {
        "audio_id": audio_id,
        "final_result_version": "v2",
        "status": "success",

        # UI が最上位を参照できるように title を持たせる
        "title": final_title,

        "audio_info": {
            "audio_id": audio_id,
            "segment_count": segment_count,
            "estimated_total_duration_sec": round(total_duration, 4),
            "sample_rate": sample_rate,
        },

        "space_analysis": {
            "space_taxonomy_version": similarity.get("space_taxonomy_version", "v1"),
            "global_space_scores": similarity.get("global_space_scores", {}),
            "top_space_similarity": similarity.get("top_space", ""),
            "final_space_label": judgement.get("final_space_label", ""),
            "confidence": judgement.get("confidence", 0.0),
            "attributes": judgement.get("attributes", {}),
            "reason": judgement.get("reason", []),
            "timeline": judgement.get("timeline", []),
            "timeline_summary": judgement.get("timeline_summary", []),
        },

        "scene_interpretation": {
            "scene_title": scene.get("scene_title", ""),
            "scene_summary": scene.get("scene_summary", ""),
            "mood_tags": scene.get("mood_tags", []),
            "environment_tags": scene.get("environment_tags", []),
            "timeline_summary": scene.get("timeline_summary", []),
            "visual_hints": scene.get("visual_hints", []),
        },

        "onomatopoeia": {
            "primary_onomatopoeia": onomatopoeia.get("primary_onomatopoeia", ""),
            "secondary_candidates": onomatopoeia.get("secondary_candidates", []),
            "intensity": onomatopoeia.get("intensity", ""),
            "style_hint": onomatopoeia.get("style_hint", ""),
            "placement_hint": onomatopoeia.get("placement_hint", ""),
            "segment_onomatopoeia": onomatopoeia.get("segment_onomatopoeia", []),
            "summary": onomatopoeia.get("summary", ""),
        },

        "manga_prompt": {
            # 旧UI互換用。manga_titleも必ず埋める
            "manga_title": final_title,

            # 新しい 08_manga_prompt.json では title が正
            "title": manga_prompt.get("title", final_title),

            "scene_match": manga_prompt.get("scene_match", ""),
            "confidence": manga_prompt.get("confidence", 0.0),
            "panel_count": manga_prompt.get("panel_count", 0),
            "panel_plan": manga_prompt.get("panel_plan", []),
            "positive_prompt": manga_prompt.get("positive_prompt", ""),
            "negative_prompt": manga_prompt.get("negative_prompt", ""),
            "onomatopoeia_layout": manga_prompt.get("onomatopoeia_layout", []),
            "direction_notes": manga_prompt.get("direction_notes", []),

            # デバッグしやすいように持たせる
            "llm": manga_prompt.get("llm", {}),
            "source_summary": manga_prompt.get("source_summary", {}),
        },

        "source_files": {
            "features": str(features_path),
            "space_similarity": str(similarity_path),
            "space_judgement": str(judgement_path),
            "scene_interpretation": str(scene_path),
            "onomatopoeia": str(onomatopoeia_path),
            "manga_prompt": str(manga_prompt_path),
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"[final_result] saved -> {output_path}")


if __name__ == "__main__":
    run("478db4fea37e")