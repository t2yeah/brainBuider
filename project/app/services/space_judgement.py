#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
space_judgement.py

役割:
- 新しい 05_space_similarity.json を読み込む
- 旧 06_space_judgement.json 互換フォーマットで出力する
- 判定ロジックを改善し、曖昧判定に落ちすぎないようにする
- 判定理由をわかりやすくする

想定入力:
  /home/team-009/project/data/results/<audio_id>/05_space_similarity.json

想定出力:
  /home/team-009/project/data/results/<audio_id>/06_space_judgement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# パス設定
# =========================================================
PROJECT_ROOT = Path("/home/team-009/project")
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# =========================================================
# ルール設定
# =========================================================

# 大分類
SPACE_FAMILY_MAP = {
    "dense_forest": "nature",
    "river_side": "nature",
    "open_field": "nature",
    "mountain_slope": "nature",
    "coast": "nature",
    "waterside": "nature",
    "forest": "nature",
    "open_nature": "nature",
    "mountain": "nature",
    "park": "nature",

    "urban_street": "urban",
    "residential_area": "urban",
    "shopping_area": "urban",
    "station_platform": "urban",
    "roadside_transport": "urban",
    "transport_space": "urban",

    "indoor_room": "indoor",
    "public_indoor": "indoor",
    "school_classroom": "indoor",
    "office_space": "indoor",
    "restaurant_cafe": "indoor",

    "industrial_space": "industrial",
    "construction_site": "industrial",

    "music_like": "other",
    "human_activity": "other",
}

# 旧版より狭める
# 「gapだけで mixed」にしないため、かなり保守的に使う
MIXED_GAP_THRESHOLD = 0.02
STRONG_GAP_THRESHOLD = 0.05

# セグメント安定性
STABLE_RATIO_THRESHOLD = 0.75
VERY_STABLE_RATIO_THRESHOLD = 0.90


# =========================================================
# 基本ユーティリティ
# =========================================================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)


def round4(x: float) -> float:
    return round(float(x), 4)


def get_family(space_id: Optional[str]) -> str:
    if not space_id:
        return "mixed"
    return SPACE_FAMILY_MAP.get(space_id, "mixed")


def format_segment_id(seg_id: Any) -> str:
    try:
        i = int(seg_id)
        return f"segment_{i:03d}"
    except Exception:
        return str(seg_id)


# =========================================================
# 入力正規化
# =========================================================

def extract_global_scores(similarity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    新フォーマットの `space_scores` を取得。
    念のため旧フォーマット `global_space_scores` にもフォールバック。
    """
    if isinstance(similarity.get("space_scores"), list):
        return similarity["space_scores"]

    if isinstance(similarity.get("global_space_scores"), list):
        out = []
        for row in similarity["global_space_scores"]:
            space_id = row.get("space_id") or row.get("label") or "unknown"
            out.append({
                "space_id": space_id,
                "label": row.get("label", space_id),
                "mean_score": float(row.get("mean_score", row.get("score", 0.0))),
                "max_score": float(row.get("max_score", row.get("score", 0.0))),
            })
        return out

    raise ValueError("No space_scores or global_space_scores found in similarity result.")


def extract_segments(similarity: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = similarity.get("segment_space_scores")
    if not isinstance(segments, list) or not segments:
        return []
    return segments


# =========================================================
# スコア解釈
# =========================================================

def sort_global_scores(global_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        global_scores,
        key=lambda x: float(x.get("mean_score", 0.0)),
        reverse=True,
    )


def get_top2_global(global_scores: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    ordered = sort_global_scores(global_scores)
    if not ordered:
        return {"space_id": "unknown", "mean_score": 0.0}, None
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) >= 2 else None
    return top1, top2


def get_top2_segment_scores(scores: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not scores:
        return None, None
    ordered = sorted(
        scores,
        key=lambda x: float(x.get("final_score", x.get("score", 0.0))),
        reverse=True,
    )
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) >= 2 else None
    return top1, top2


def compute_dominant_ratio(segments: List[Dict[str, Any]], primary_space: str) -> float:
    if not segments:
        return 0.0
    hit = sum(1 for seg in segments if seg.get("top_space") == primary_space)
    return hit / len(segments)


def detect_transition(segments: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    セグメントの top_space が変わった箇所を抽出
    """
    if len(segments) <= 1:
        return False, []

    changes = []
    prev = segments[0]
    for cur in segments[1:]:
        prev_space = prev.get("top_space")
        cur_space = cur.get("top_space")
        if prev_space != cur_space:
            changes.append({
                "from_segment_id": format_segment_id(prev.get("segment_id")),
                "to_segment_id": format_segment_id(cur.get("segment_id")),
                "from_space": prev_space,
                "to_space": cur_space,
            })
        prev = cur

    return len(changes) > 0, changes


def build_timeline(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    旧 06 の timeline 互換
    """
    out = []

    for seg in segments:
        seg_scores = seg.get("scores", [])
        top1, top2 = get_top2_segment_scores(seg_scores)

        if top1 is None:
            top_space = seg.get("top_space", "unknown")
            top_score = float(seg.get("top_score", 0.0))
            secondary_space = None
            secondary_score = 0.0
        else:
            top_space = top1.get("space_id", seg.get("top_space", "unknown"))
            top_score = float(top1.get("final_score", seg.get("top_score", 0.0)))
            secondary_space = top2.get("space_id") if top2 else None
            secondary_score = float(top2.get("final_score", 0.0)) if top2 else 0.0

        out.append({
            "segment_id": format_segment_id(seg.get("segment_id")),
            "top_space": top_space,
            "top_score": round4(top_score),
            "secondary_space": secondary_space,
            "secondary_score": round4(secondary_score),
        })

    return out


# =========================================================
# 追加ユーティリティ
# =========================================================

def get_stability_level(dominant_ratio: float) -> str:
    if dominant_ratio >= 0.90:
        return "very_stable"
    if dominant_ratio >= 0.75:
        return "stable"
    return "unstable"


def get_gap_level(score_gap: float) -> str:
    if score_gap < 0.02:
        return "close"
    if score_gap < 0.05:
        return "medium"
    return "clear"


# =========================================================
# 判定ロジック
# =========================================================

def decide_final_space_label(
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
) -> str:
    """
    final_space_label を返す。

    方針:
    1. 時系列の安定性を優先
    2. 次に gap を見る
    3. 本当に曖昧なときだけ mixed_ambiguous にする
    """
    if secondary_space is None:
        return primary_space

    # 1. 全体がかなり安定していて遷移がなければ primary 優先
    if dominant_ratio >= VERY_STABLE_RATIO_THRESHOLD and not transition_detected:
        return primary_space

    # 2. そこそこ安定 + gap も最低限あれば primary
    if dominant_ratio >= STABLE_RATIO_THRESHOLD and score_gap >= 0.03:
        return primary_space

    # 3. 遷移があり、しかも十分安定していないなら mixed
    if transition_detected and dominant_ratio < 0.85:
        return "mixed_ambiguous"

    # 4. gap が極端に近く、なおかつ安定性も弱いなら mixed
    if score_gap < MIXED_GAP_THRESHOLD and dominant_ratio < VERY_STABLE_RATIO_THRESHOLD:
        return "mixed_ambiguous"

    # 5. 迷う場合は primary を優先
    return primary_space


def decide_environment_family(
    final_space_label: str,
    primary_space: str,
    secondary_space: Optional[str],
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
) -> str:
    if final_space_label == "mixed_ambiguous":
        return "mixed"

    primary_family = get_family(primary_space)
    secondary_family = get_family(secondary_space) if secondary_space else primary_family

    # final は primary でも、family だけは混合表現の方が自然な場合を残す
    if (
        primary_family != secondary_family
        and score_gap < 0.02
        and dominant_ratio < VERY_STABLE_RATIO_THRESHOLD
        and transition_detected
    ):
        return "mixed"

    return primary_family


def compute_confidence(
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
) -> float:
    """
    0-1 の confidence を返す。
    """
    score_strength = primary_score
    gap_strength = min(max(score_gap / 0.20, 0.0), 1.0)
    stability_strength = min(max(dominant_ratio, 0.0), 1.0)
    transition_penalty = 0.08 if transition_detected else 0.0

    conf = (
        score_strength * 0.40
        + gap_strength * 0.30
        + stability_strength * 0.30
        - transition_penalty
    )

    conf = max(0.0, min(conf, 1.0))
    return round4(conf)


def build_reason(
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    final_space_label: str,
    dominant_ratio: float,
    transition_detected: bool,
) -> List[str]:
    reason = []

    reason.append(f"最上位候補は {primary_space} ({primary_score:.4f})")

    if secondary_space is not None:
        reason.append(f"次点候補は {secondary_space} ({secondary_score:.4f})")
        reason.append(f"上位2候補のスコア差は {score_gap:.4f}")

    reason.append(f"主空間のセグメント占有率は {dominant_ratio * 100.0:.1f}% です。")

    if transition_detected:
        reason.append("セグメント推移上、途中で空間傾向の変化が見られました。")
    else:
        reason.append("セグメント推移上、明確な空間遷移は見られませんでした。")

    if final_space_label == "mixed_ambiguous":
        reason.append("上位候補が近接しており、主空間を1つに確定するには根拠が不足しているため、混合または曖昧と判断しました。")
    else:
        reason.append(f"時系列の安定性を優先し、最終的に {primary_space} と判断しました。")

    return reason


def build_timeline_summary(
    timeline: List[Dict[str, Any]],
    primary_space: str,
    dominant_ratio: float,
    transition_detected: bool,
    transition_points: List[Dict[str, Any]],
) -> List[str]:
    if not timeline:
        return ["セグメント情報がないため、時系列の空間傾向は判定できませんでした。"]

    summaries = []

    if not transition_detected:
        summaries.append(f"全セグメントを通じて {primary_space} 傾向が継続しています。")
        summaries.append(f"{primary_space} のセグメント占有率は {dominant_ratio * 100.0:.1f}% です。")
        return summaries

    summaries.append(
        f"主空間は全体として {primary_space} ですが、時系列上で空間傾向の変化が見られます。"
    )

    for tp in transition_points:
        summaries.append(
            f"{tp['from_segment_id']} → {tp['to_segment_id']} で "
            f"{tp['from_space']} から {tp['to_space']} へ遷移しています。"
        )

    summaries.append(f"{primary_space} のセグメント占有率は {dominant_ratio * 100.0:.1f}% です。")
    return summaries


# =========================================================
# 本体
# =========================================================

def build_judgement(similarity: Dict[str, Any]) -> Dict[str, Any]:
    audio_id = similarity.get("audio_id", "unknown")
    judgement_version = similarity.get("space_taxonomy_version", "v3_imagable_general")

    global_scores = extract_global_scores(similarity)
    segments = extract_segments(similarity)

    top1, top2 = get_top2_global(global_scores)

    primary_space = top1.get("space_id", "unknown")
    secondary_space = top2.get("space_id") if top2 else None

    primary_score = float(top1.get("mean_score", 0.0))
    secondary_score = float(top2.get("mean_score", 0.0)) if top2 else 0.0
    score_gap = primary_score - secondary_score

    dominant_ratio = compute_dominant_ratio(segments, primary_space)
    transition_detected, transition_points = detect_transition(segments)

    final_space_label = decide_final_space_label(
        primary_space=primary_space,
        secondary_space=secondary_space,
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
    )

    environment_family = decide_environment_family(
        final_space_label=final_space_label,
        primary_space=primary_space,
        secondary_space=secondary_space,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
    )

    confidence = compute_confidence(
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
    )

    all_global_scores = {
        row.get("space_id", f"space_{i}"): round4(float(row.get("mean_score", 0.0)))
        for i, row in enumerate(sort_global_scores(global_scores))
    }

    timeline = build_timeline(segments)

    reason = build_reason(
        primary_space=primary_space,
        secondary_space=secondary_space,
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        final_space_label=final_space_label,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
    )

    timeline_summary = build_timeline_summary(
        timeline=timeline,
        primary_space=primary_space,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_points=transition_points,
    )

    result = {
        "audio_id": audio_id,
        "judgement_version": judgement_version,
        "final_space_label": final_space_label,
        "confidence": confidence,
        "environment_family": environment_family,
        "attributes": {
            "primary_space": primary_space,
            "secondary_space": secondary_space,
            "primary_score": round4(primary_score),
            "secondary_score": round4(secondary_score),
            "score_gap": round4(score_gap),
            "all_global_scores": all_global_scores,
        },
        "reason": reason,
        "timeline": timeline,
        "timeline_summary": timeline_summary,
        "extras": {
            "dominant_ratio": round4(dominant_ratio),
            "transition_detected": transition_detected,
            "transition_points": transition_points,
            "segment_count": len(segments),
            "stability_level": get_stability_level(dominant_ratio),
            "gap_level": get_gap_level(score_gap),
            "decision_basis": (
                "timeline_priority"
                if dominant_ratio >= VERY_STABLE_RATIO_THRESHOLD and not transition_detected
                else "balanced"
                if final_space_label != "mixed_ambiguous"
                else "ambiguity_priority"
            ),
        },
    }

    return result


def get_similarity_path(audio_id: str) -> Path:
    return RESULTS_DIR / audio_id / "05_space_similarity.json"


def get_output_path(audio_id: str) -> Path:
    return RESULTS_DIR / audio_id / "06_space_judgement.json"


def run(audio_id: str, pretty: bool = True) -> Dict[str, Any]:
    similarity_path = get_similarity_path(audio_id)
    if not similarity_path.exists():
        raise FileNotFoundError(f"05_space_similarity.json not found: {similarity_path}")

    similarity = load_json(similarity_path)
    result = build_judgement(similarity)

    output_path = get_output_path(audio_id)
    save_json(output_path, result, pretty=pretty)

    return result


def run_space_judgement(audio_id: str, pretty: bool = True) -> Dict[str, Any]:
    return run(audio_id, pretty=pretty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True, help="audio_id")
    parser.add_argument("--pretty", action="store_true", help="pretty print JSON")
    args = parser.parse_args()

    result = run(args.audio_id, pretty=args.pretty)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()