#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
space_judgement.py

役割:
- 05_space_similarity.json を読み込む
- 06_space_judgement.json 互換フォーマットで出力する
- 時系列安定性 / 全体スコア / セグメント推移を総合して最終空間を決定する
- 必要時のみ LLM semantic review を使って、安定した誤判定を抑制する
- 例外時も追跡しやすく、cleanup を finally で必ず実行する

想定入力:
  /home/team-009/project/data/results/<audio_id>/05_space_similarity.json

任意入力:
  /home/team-009/project/data/results/<audio_id>/07_scene_interpretation.json

想定出力:
  /home/team-009/project/data/results/<audio_id>/06_space_judgement.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:
    torch = None


# =========================================================
# パス設定
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# =========================================================
# 空間ファミリー
# =========================================================
SPACE_FAMILY_MAP: Dict[str, str] = {
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


# =========================================================
# 閾値
# =========================================================
VERY_STABLE_RATIO_THRESHOLD = 0.90
STABLE_RATIO_THRESHOLD = 0.75
LOW_GAP_THRESHOLD = 0.02
MID_GAP_THRESHOLD = 0.05
HIGH_TRANSITION_RATIO_THRESHOLD = 0.20

# LLM semantic review 条件
LLM_REVIEW_DOMINANT_RATIO_THRESHOLD = 0.95
LLM_REVIEW_GAP_THRESHOLD = 0.10
LLM_REVIEW_CONFIDENCE_THRESHOLD = 0.72
LLM_REVIEW_MAX_CANDIDATES = 5
LLM_REVIEW_MAX_SEGMENT_HINTS = 5

# LLM設定
DEFAULT_LLM_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_TOP_P = 0.8
DEFAULT_LLM_MAX_NEW_TOKENS = 512


# =========================================================
# 基本ユーティリティ
# =========================================================
def round4(x: float) -> float:
    return round(float(x), 4)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def save_json(path: Path, data: Dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)


def get_family(space_id: Optional[str]) -> str:
    if not space_id:
        return "mixed"
    return SPACE_FAMILY_MAP.get(space_id, "mixed")


def format_segment_id(seg_id: Any) -> str:
    if isinstance(seg_id, str) and seg_id.startswith("segment_"):
        return seg_id
    try:
        i = int(seg_id)
        return f"segment_{i:03d}"
    except Exception:
        return str(seg_id)


def cleanup_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass

    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


# =========================================================
# パス
# =========================================================
def get_similarity_path(audio_id: str) -> Path:
    return RESULTS_DIR / audio_id / "05_space_similarity.json"


def get_scene_path(audio_id: str) -> Path:
    return RESULTS_DIR / audio_id / "07_scene_interpretation.json"


def get_output_path(audio_id: str) -> Path:
    return RESULTS_DIR / audio_id / "06_space_judgement.json"


# =========================================================
# 入力正規化
# =========================================================
def extract_global_scores(similarity: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(similarity.get("space_scores"), list):
        out: List[Dict[str, Any]] = []
        for row in similarity["space_scores"]:
            if not isinstance(row, dict):
                continue
            space_id = row.get("space_id") or row.get("label") or "unknown"
            out.append({
                "space_id": str(space_id),
                "label": row.get("label", str(space_id)),
                "mean_score": safe_float(row.get("mean_score", row.get("score", 0.0))),
                "max_score": safe_float(row.get("max_score", row.get("score", 0.0))),
            })
        if out:
            return out

    if isinstance(similarity.get("global_space_scores"), list):
        out = []
        for row in similarity["global_space_scores"]:
            if not isinstance(row, dict):
                continue
            space_id = row.get("space_id") or row.get("label") or "unknown"
            out.append({
                "space_id": str(space_id),
                "label": row.get("label", str(space_id)),
                "mean_score": safe_float(row.get("mean_score", row.get("score", 0.0))),
                "max_score": safe_float(row.get("max_score", row.get("score", 0.0))),
            })
        if out:
            return out

    raise ValueError("No usable space_scores or global_space_scores found in similarity result.")


def extract_segments(similarity: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = similarity.get("segment_space_scores")
    if not isinstance(segments, list):
        return []
    return [seg for seg in segments if isinstance(seg, dict)]


def load_optional_scene(audio_id: str) -> Optional[Dict[str, Any]]:
    path = get_scene_path(audio_id)
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def sort_global_scores(global_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        global_scores,
        key=lambda x: safe_float(x.get("mean_score", 0.0)),
        reverse=True,
    )


def get_top2_global(global_scores: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    ordered = sort_global_scores(global_scores)
    if not ordered:
        return {"space_id": "unknown", "label": "unknown", "mean_score": 0.0, "max_score": 0.0}, None
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) >= 2 else None
    return top1, top2


def get_top2_segment_scores(scores: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    usable = [s for s in scores if isinstance(s, dict)]
    if not usable:
        return None, None

    ordered = sorted(
        usable,
        key=lambda x: safe_float(x.get("final_score", x.get("score", 0.0))),
        reverse=True,
    )
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) >= 2 else None
    return top1, top2


# =========================================================
# 時系列解析
# =========================================================
def compute_dominant_ratio(segments: List[Dict[str, Any]], primary_space: str) -> float:
    if not segments:
        return 0.0

    hit = 0
    total = 0
    for seg in segments:
        top_space = seg.get("top_space")
        if top_space is None:
            continue
        total += 1
        if top_space == primary_space:
            hit += 1

    return (hit / total) if total > 0 else 0.0


def detect_transition(segments: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    usable = [seg for seg in segments if seg.get("top_space") is not None]
    if len(usable) <= 1:
        return False, []

    changes: List[Dict[str, Any]] = []
    prev = usable[0]
    for cur in usable[1:]:
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


def compute_transition_ratio(segments: List[Dict[str, Any]], transition_points: List[Dict[str, Any]]) -> float:
    usable = [seg for seg in segments if seg.get("top_space") is not None]
    if len(usable) <= 1:
        return 0.0
    return len(transition_points) / max(len(usable) - 1, 1)


def build_timeline(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for seg in segments:
        seg_scores = seg.get("scores", [])
        if not isinstance(seg_scores, list):
            seg_scores = []

        top1, top2 = get_top2_segment_scores(seg_scores)

        if top1 is None:
            top_space = seg.get("top_space", "unknown")
            top_score = safe_float(seg.get("top_score", 0.0))
            secondary_space = None
            secondary_score = 0.0
        else:
            top_space = top1.get("space_id", seg.get("top_space", "unknown"))
            top_score = safe_float(top1.get("final_score", seg.get("top_score", 0.0)))
            secondary_space = top2.get("space_id") if top2 else None
            secondary_score = safe_float(top2.get("final_score", 0.0)) if top2 else 0.0

        out.append({
            "segment_id": format_segment_id(seg.get("segment_id")),
            "top_space": top_space,
            "top_score": round4(top_score),
            "secondary_space": secondary_space,
            "secondary_score": round4(secondary_score),
        })

    return out


# =========================================================
# 指標
# =========================================================
def get_stability_level(dominant_ratio: float) -> str:
    if dominant_ratio >= VERY_STABLE_RATIO_THRESHOLD:
        return "very_stable"
    if dominant_ratio >= STABLE_RATIO_THRESHOLD:
        return "stable"
    return "unstable"


def get_gap_level(score_gap: float) -> str:
    if score_gap < LOW_GAP_THRESHOLD:
        return "close"
    if score_gap < MID_GAP_THRESHOLD:
        return "medium"
    return "clear"


def compute_confidence(
    primary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
    segment_count: int,
) -> float:
    score_strength = min(max(primary_score, 0.0), 1.0)
    gap_strength = min(max(score_gap / 0.20, 0.0), 1.0)
    stability_strength = min(max(dominant_ratio, 0.0), 1.0)

    transition_penalty = 0.0
    if transition_detected:
        transition_penalty = min(0.12, transition_ratio * 0.20)

    low_segment_penalty = 0.05 if 0 < segment_count < 3 else 0.0

    conf = (
        score_strength * 0.40
        + gap_strength * 0.30
        + stability_strength * 0.30
        - transition_penalty
        - low_segment_penalty
    )

    conf = max(0.0, min(conf, 1.0))
    return round4(conf)


# =========================================================
# provisional 判定
# =========================================================
def decide_final_space_label(
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
    segment_count: int,
) -> str:
    if not secondary_space:
        return primary_space

    if segment_count == 0:
        return primary_space

    if dominant_ratio >= VERY_STABLE_RATIO_THRESHOLD and not transition_detected:
        return primary_space

    if dominant_ratio >= STABLE_RATIO_THRESHOLD and score_gap >= LOW_GAP_THRESHOLD:
        return primary_space

    highly_ambiguous = (
        score_gap < LOW_GAP_THRESHOLD
        and dominant_ratio < STABLE_RATIO_THRESHOLD
    )

    temporally_unstable = (
        transition_detected
        and transition_ratio >= HIGH_TRANSITION_RATIO_THRESHOLD
        and dominant_ratio < 0.85
    )

    if highly_ambiguous or temporally_unstable:
        return "mixed_ambiguous"

    return primary_space


def decide_environment_family(
    final_space_label: str,
    primary_space: str,
    secondary_space: Optional[str],
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
) -> str:
    if final_space_label == "mixed_ambiguous":
        return "mixed"

    primary_family = get_family(primary_space)
    secondary_family = get_family(secondary_space) if secondary_space else primary_family

    if (
        primary_family != secondary_family
        and score_gap < LOW_GAP_THRESHOLD
        and dominant_ratio < VERY_STABLE_RATIO_THRESHOLD
        and transition_detected
        and transition_ratio >= HIGH_TRANSITION_RATIO_THRESHOLD
    ):
        return "mixed"

    return primary_family


# =========================================================
# LLM semantic review
# =========================================================
def try_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        return AutoTokenizer, AutoModelForCausalLM
    except Exception:
        return None, None


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON", text, re.DOTALL)
    if m:
        return m.group(1)

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1)

    return None


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    block = extract_json_block(text)
    if not block:
        return None
    try:
        data = json.loads(block)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_local_llm():
    AutoTokenizer, AutoModelForCausalLM = try_import_transformers()
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not available")

    model_id = os.environ.get("SPACE_JUDGEMENT_LLM_MODEL", DEFAULT_LLM_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {}
    if torch is not None:
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return tokenizer, model, model_id


def call_local_llm_json(system_prompt: str, user_prompt: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "used": False,
        "status": "not_run",
        "model": os.environ.get("SPACE_JUDGEMENT_LLM_MODEL", DEFAULT_LLM_MODEL),
        "reason": "",
        "raw_preview": "",
    }

    tokenizer = None
    model = None

    try:
        tokenizer, model, resolved_model_id = load_local_llm()
        meta["used"] = True
        meta["model"] = resolved_model_id

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt")

        if torch is not None and hasattr(model, "device"):
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except Exception:
                pass

        max_new_tokens = int(os.environ.get("SPACE_JUDGEMENT_LLM_MAX_NEW_TOKENS", DEFAULT_LLM_MAX_NEW_TOKENS))
        temperature = float(os.environ.get("SPACE_JUDGEMENT_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE))
        top_p = float(os.environ.get("SPACE_JUDGEMENT_LLM_TOP_P", DEFAULT_LLM_TOP_P))

        with torch.no_grad() if torch is not None else nullcontext():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        meta["raw_preview"] = raw_text[:1200]

        parsed = parse_json_loose(raw_text)
        if parsed is None:
            meta["status"] = "parse_failed"
            meta["reason"] = "llm output could not be parsed as JSON"
            return None, meta

        meta["status"] = "success"
        meta["reason"] = "llm semantic review succeeded"
        return parsed, meta

    except Exception as e:
        meta["status"] = "error"
        meta["reason"] = f"{type(e).__name__}: {e}"
        return None, meta

    finally:
        try:
            del tokenizer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        cleanup_memory()


def collect_audio_event_evidence(
    similarity: Dict[str, Any],
    scene: Optional[Dict[str, Any]],
) -> List[str]:
    """
    LLMに渡す強い証拠だけを集める。
    全部は渡さない。
    """
    candidates: List[str] = []

    sim_tags = similarity.get("audio_event_tags", [])
    if isinstance(sim_tags, list):
        for x in sim_tags:
            if isinstance(x, str):
                candidates.append(x)

    scene_tags = []
    if scene and isinstance(scene.get("audio_event_tags"), list):
        scene_tags = [x for x in scene["audio_event_tags"] if isinstance(x, str)]
        candidates.extend(scene_tags)

    # 類似順を保ちつつ重複除去
    uniq: List[str] = []
    seen = set()
    for x in candidates:
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    # 重要度の高い少数だけ
    return uniq[:6]


def collect_segment_evidence(scene: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not scene:
        return []

    hints = scene.get("segment_audio_hints", [])
    if not isinstance(hints, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in hints[:LLM_REVIEW_MAX_SEGMENT_HINTS]:
        if not isinstance(item, dict):
            continue
        raw_labels = item.get("raw_labels", [])
        if not isinstance(raw_labels, list):
            raw_labels = []
        out.append({
            "segment_id": item.get("segment_id"),
            "description": item.get("description"),
            "raw_labels": [x for x in raw_labels if isinstance(x, str)][:5],
        })
    return out


def build_llm_review_payload(
    similarity: Dict[str, Any],
    scene: Optional[Dict[str, Any]],
    global_scores: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
    provisional_final_space_label: str,
    provisional_confidence: float,
) -> Dict[str, Any]:
    ordered_scores = sort_global_scores(global_scores)
    candidate_spaces = [
        row.get("space_id")
        for row in ordered_scores[:LLM_REVIEW_MAX_CANDIDATES]
        if row.get("space_id")
    ]

    if provisional_final_space_label != "mixed_ambiguous" and provisional_final_space_label not in candidate_spaces:
        candidate_spaces.append(provisional_final_space_label)
    if "mixed_ambiguous" not in candidate_spaces:
        candidate_spaces.append("mixed_ambiguous")

    representative_timeline = []
    built_timeline = build_timeline(segments)
    for row in built_timeline[:6]:
        representative_timeline.append({
            "segment_id": row["segment_id"],
            "top_space": row["top_space"],
            "top_score": row["top_score"],
            "secondary_space": row["secondary_space"],
            "secondary_score": row["secondary_score"],
        })

    audio_events_top = collect_audio_event_evidence(similarity, scene)
    segments_sample = collect_segment_evidence(scene)

    # provisional を疑うための監査観点を明示
    audit_focus = {
        "stable_but_maybe_wrong": bool(dominant_ratio >= 0.95 and score_gap < 0.10),
        "contains_large_water_signals": any(
            x in audio_events_top
            for x in ["Ocean", "Waves, surf", "Boat, Water vehicle"]
        ),
        "contains_transport_or_vehicle_signals": any(
            x in audio_events_top
            for x in ["Vehicle", "Boat, Water vehicle"]
        ),
        "needs_conservative_decision": True,
    }

    payload = {
        "audio_id": similarity.get("audio_id", "unknown"),
        "task": "semantic_audit",
        "question": "Is the provisional label semantically safe, or should it remain ambiguous?",
        "provisional": {
            "final_space_label": provisional_final_space_label,
            "confidence": round4(provisional_confidence),
        },
        "candidates": candidate_spaces,
        "evidence": {
            "global_top_scores": {
                str(row.get("space_id", f"space_{i}")): round4(safe_float(row.get("mean_score", 0.0)))
                for i, row in enumerate(ordered_scores[:LLM_REVIEW_MAX_CANDIDATES])
            },
            "primary_space": primary_space,
            "secondary_space": secondary_space,
            "primary_score": round4(primary_score),
            "secondary_score": round4(secondary_score),
            "score_gap": round4(score_gap),
            "dominant_ratio": round4(dominant_ratio),
            "transition_detected": transition_detected,
            "transition_ratio": round4(transition_ratio),
            "audio_events_top": audio_events_top,
            "segments_sample": segments_sample,
            "timeline_sample": representative_timeline,
            "scene_title": scene.get("scene_title") if scene else None,
            "scene_summary": scene.get("scene_summary") if scene else None,
            "audio_event_summary": scene.get("audio_event_summary") if scene else None,
        },
        "audit_focus": audit_focus,
        "decision_policy": {
            "prefer_provisional_only_if_semantically_safe": True,
            "choose_mixed_ambiguous_if_evidence_is_stable_but_semantically_conflicting": True,
            "do_not_overcorrect_to_secondary_without_clear_reason": True,
        },
    }
    return payload




def build_llm_system_prompt() -> str:
    return (
        "You are a strict semantic auditor for audio scene classification.\n"
        "Your job is to challenge the provisional label, not to casually confirm it.\n"
        "You must choose only from the provided candidates.\n"
        "If the provisional label is temporally stable but semantically broad, weak, or questionable, prefer mixed_ambiguous.\n"
        "Do not invent a new label.\n"
        "Do not reward temporal stability alone.\n"
        "Temporal consistency is useful, but it does not guarantee semantic correctness.\n"
        "If the evidence includes Ocean, Waves, surf, or Boat, Water vehicle, do not keep river_side unless the evidence clearly supports a river-like scene.\n"
        "If semantic evidence is broad water, open water, coastal, or otherwise not clearly river-specific, prefer mixed_ambiguous.\n"
        "If you keep the provisional label, you must justify why it is semantically safe.\n"
        "If that justification is weak, choose mixed_ambiguous.\n"
        "Output JSON only.\n"
        "Output format:\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "final_space_label": "one_of_candidates",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short Japanese reason"\n'
        "}\n"
        "END_JSON"
    )
    

def build_llm_user_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Audit the provisional audio-space judgement.\n"
        "Rules:\n"
        "- final_space_label must be one of candidates.\n"
        "- Do not create any new label.\n"
        "- Treat the provisional label as suspicious unless it is clearly semantically safe.\n"
        "- Do not rely on dominant_ratio alone.\n"
        "- If the scene is temporally stable but semantic evidence is still broad or conflicting, prefer mixed_ambiguous.\n"
        "- If large-water evidence exists but river-specific evidence is weak, prefer mixed_ambiguous over river_side.\n"
        "- reason must be short, concrete, and non-empty Japanese.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def should_run_llm_review(
    provisional_final_space_label: str,
    dominant_ratio: float,
    score_gap: float,
    provisional_confidence: float,
    transition_detected: bool,
) -> bool:
    if provisional_final_space_label == "mixed_ambiguous":
        return True

    if dominant_ratio >= 0.95 and score_gap < 0.10:
        return True

    if provisional_confidence < LLM_REVIEW_CONFIDENCE_THRESHOLD:
        return True

    if transition_detected:
        return True

    return False
    

def validate_llm_review_result(
    llm_data: Dict[str, Any],
    candidate_spaces: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(llm_data, dict):
        return None

    label = llm_data.get("final_space_label")
    confidence = safe_float(llm_data.get("confidence", -1.0), -1.0)
    reason = llm_data.get("reason", "")

    if not isinstance(label, str):
        return None
    if label not in candidate_spaces:
        return None
    if not (0.0 <= confidence <= 1.0):
        return None

    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = str(reason)

    reason = reason.strip()

    # reason は出力必須ではないが、品質チェックには使う
    # 空や短すぎる場合は「実質何も監査していない」とみなして不採用
    if len(reason) < 10:
        return None

    return {
        "final_space_label": label,
        "confidence": round4(confidence),
        "reason": reason[:300],
    }


def maybe_apply_llm_review(
    similarity: Dict[str, Any],
    scene: Optional[Dict[str, Any]],
    global_scores: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
    provisional_final_space_label: str,
    provisional_confidence: float,
) -> Tuple[Optional[str], Optional[float]]:
    if not should_run_llm_review(
        provisional_final_space_label=provisional_final_space_label,
        dominant_ratio=dominant_ratio,
        score_gap=score_gap,
        provisional_confidence=provisional_confidence,
        transition_detected=transition_detected,
    ):
        return provisional_final_space_label, provisional_confidence

    payload = build_llm_review_payload(
        similarity=similarity,
        scene=scene,
        global_scores=global_scores,
        segments=segments,
        primary_space=primary_space,
        secondary_space=secondary_space,
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_ratio=transition_ratio,
        provisional_final_space_label=provisional_final_space_label,
        provisional_confidence=provisional_confidence,
    )

    llm_data, _meta = call_local_llm_json(
        system_prompt=build_llm_system_prompt(),
        user_prompt=build_llm_user_prompt(payload),
    )

    if not llm_data:
       return None, None

    valid = validate_llm_review_result(llm_data, payload["candidates"])
    if not valid:
        return None, None

    return valid["final_space_label"], valid["confidence"]


# =========================================================
# 理由生成
# =========================================================
def build_reason(
    primary_space: str,
    secondary_space: Optional[str],
    primary_score: float,
    secondary_score: float,
    score_gap: float,
    final_space_label: str,
    dominant_ratio: float,
    transition_detected: bool,
    transition_ratio: float,
) -> List[str]:
    reason: List[str] = []

    reason.append(f"最上位候補は {primary_space} ({primary_score:.4f})")

    if secondary_space is not None:
        reason.append(f"次点候補は {secondary_space} ({secondary_score:.4f})")
        reason.append(f"上位2候補のスコア差は {score_gap:.4f}")

    reason.append(f"主空間のセグメント占有率は {dominant_ratio * 100.0:.1f}% です。")

    if transition_detected:
        reason.append(
            f"セグメント推移上、途中で空間傾向の変化が見られました（遷移率 {transition_ratio * 100.0:.1f}%）。"
        )
    else:
        reason.append("セグメント推移上、明確な空間遷移は見られませんでした。")

    if final_space_label == "mixed_ambiguous":
        reason.append(
            "全体スコアや時系列は一定の傾向を示すものの、意味的に単一空間へ強く確定する根拠が不足したため、混合または曖昧と判断しました。"
        )
    else:
        reason.append(f"全体スコアと時系列安定性を総合し、最終的に {final_space_label} と判断しました。")

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

    summaries: List[str] = []

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

    scene = load_optional_scene(audio_id)

    global_scores = extract_global_scores(similarity)
    segments = extract_segments(similarity)

    top1, top2 = get_top2_global(global_scores)

    primary_space = str(top1.get("space_id", "unknown"))
    secondary_space = str(top2.get("space_id")) if top2 and top2.get("space_id") is not None else None

    primary_score = safe_float(top1.get("mean_score", 0.0))
    secondary_score = safe_float(top2.get("mean_score", 0.0)) if top2 else 0.0
    score_gap = max(0.0, primary_score - secondary_score)

    dominant_ratio = compute_dominant_ratio(segments, primary_space)
    transition_detected, transition_points = detect_transition(segments)
    transition_ratio = compute_transition_ratio(segments, transition_points)

    provisional_final_space_label = decide_final_space_label(
        primary_space=primary_space,
        secondary_space=secondary_space,
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_ratio=transition_ratio,
        segment_count=len(segments),
    )

    provisional_confidence = compute_confidence(
        primary_score=primary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_ratio=transition_ratio,
        segment_count=len(segments),
    )
    stable_but_semantically_risky = (
        dominant_ratio >= 0.95
        and score_gap < 0.10
        and primary_space == provisional_final_space_label
    )

    # ===== LLM review（安全なフォールバック付き）=====
    llm_label, llm_conf = maybe_apply_llm_review(
        similarity=similarity,
        scene=scene,
        global_scores=global_scores,
        segments=segments,
        primary_space=primary_space,
        secondary_space=secondary_space,
        primary_score=primary_score,
        secondary_score=secondary_score,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_ratio=transition_ratio,
        provisional_final_space_label=provisional_final_space_label,
        provisional_confidence=provisional_confidence,
    )

    if llm_label is None or llm_conf is None or llm_conf <= 0.0:
        final_space_label = provisional_final_space_label
        confidence = provisional_confidence
    else:
        final_space_label = llm_label
        confidence = max(0.0, min(llm_conf, 1.0))
    # ===== ここまで =====

    environment_family = decide_environment_family(
        final_space_label=final_space_label,
        primary_space=primary_space,
        secondary_space=secondary_space,
        score_gap=score_gap,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_ratio=transition_ratio,
    )

    all_global_scores = {
        str(row.get("space_id", f"space_{i}")): round4(safe_float(row.get("mean_score", 0.0)))
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
        transition_ratio=transition_ratio,
    )

    timeline_summary = build_timeline_summary(
        timeline=timeline,
        primary_space=primary_space,
        dominant_ratio=dominant_ratio,
        transition_detected=transition_detected,
        transition_points=transition_points,
    )

    result: Dict[str, Any] = {
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
            "transition_ratio": round4(transition_ratio),
            "transition_points": transition_points,
            "segment_count": len(segments),
            "stability_level": get_stability_level(dominant_ratio),
            "gap_level": get_gap_level(score_gap),
            "decision_basis": (
                "timeline_priority"
                if (
                    dominant_ratio >= VERY_STABLE_RATIO_THRESHOLD
                    and not transition_detected
                    and final_space_label == provisional_final_space_label
                    and not stable_but_semantically_risky
                )
                else "ambiguity_priority"
                if final_space_label == "mixed_ambiguous"
                else "balanced"
            ),
        },
    }

    return result


# =========================================================
# 実行
# =========================================================
def run(audio_id: str, pretty: bool = True) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        similarity_path = get_similarity_path(audio_id)
        if not similarity_path.exists():
            raise FileNotFoundError(f"05_space_similarity.json not found: {similarity_path}")

        similarity = load_json(similarity_path)
        result = build_judgement(similarity)

        output_path = get_output_path(audio_id)
        save_json(output_path, result, pretty=pretty)

        return result
    finally:
        cleanup_memory()


def run_space_judgement(audio_id: str, pretty: bool = True) -> Dict[str, Any]:
    return run(audio_id, pretty=pretty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True, help="audio_id")
    parser.add_argument("--pretty", action="store_true", help="pretty print JSON")
    args = parser.parse_args()

    try:
        result = run(args.audio_id, pretty=args.pretty)
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    except Exception as e:
        error_payload = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    main()