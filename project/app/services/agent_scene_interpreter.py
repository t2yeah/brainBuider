from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from copy import deepcopy


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"
OUTPUT_FILENAME = "07_scene_interpretation.json"

# =========================================================
# Local LLM settings
# =========================================================
LOCAL_MODEL_NAME = os.getenv(
    "SWALLOW_LOCAL_MODEL",
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
).strip()

LOCAL_LLM_ENABLED = os.getenv("SWALLOW_LOCAL_ENABLED", "1").strip() not in {"0", "false", "False"}
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("SWALLOW_MAX_TOKENS", "1000"))
LOCAL_LLM_TEMPERATURE = float(os.getenv("SWALLOW_TEMPERATURE", "0.2"))
LOCAL_LLM_TOP_P = float(os.getenv("SWALLOW_TOP_P", "0.8"))

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

_tokenizer = None
_model = None


# =========================================================
# Memory Management
# =========================================================
def clear_gpu_memory() -> None:
    """モデル参照を外し、GC と CUDA キャッシュを解放する。"""
    global _model, _tokenizer

    print("[scene_interpreter] clearing model / tokenizer memory...")

    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None

    if _tokenizer is not None:
        try:
            del _tokenizer
        except Exception:
            pass
        _tokenizer = None

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("[scene_interpreter] memory cleared.")


# =========================================================
# JSON helpers
# =========================================================
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def maybe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def clamp_tags(tags: List[str], max_items: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in tags:
        s = str(x).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


# =========================================================
# Space / event normalization
# =========================================================
def normalize_space_label(label: str) -> str:
    mapping = {
        "dense_forest": "forest",
        "forest": "forest",
        "open_field": "field",
        "field": "field",
        "mountain_slope": "mountain",
        "mountain": "mountain",
        "river_side": "riverside",
        "riverside": "riverside",
        "roadside": "roadside",
        "residential_area": "residential",
        "urban_street": "urban",
        "urban": "urban",
        "indoor_room": "indoor",
        "indoor": "indoor",
        "unknown": "unknown",
        "mixed_ambiguous": "unknown",
    }
    key = str(label or "").strip().lower()
    return mapping.get(key, key or "unknown")


def space_to_ja(label: str) -> str:
    mapping = {
        "forest": "森",
        "field": "開けた草地",
        "mountain": "山あい",
        "riverside": "川辺",
        "roadside": "道路沿い",
        "residential": "住宅地",
        "urban": "街中",
        "indoor": "屋内",
        "unknown": "不明",
    }
    return mapping.get(normalize_space_label(label), str(label or "不明"))


def environment_family_to_tags(family: str) -> List[str]:
    fam = str(family or "").strip().lower()
    mapping = {
        "forest": ["forest", "outdoor"],
        "nature": ["outdoor", "nature"],
        "roadside": ["roadside", "outdoor"],
        "urban": ["urban", "outdoor"],
        "riverside": ["riverside", "outdoor"],
        "indoor": ["indoor"],
        "mixed": ["outdoor"],
    }
    return mapping.get(fam, ["outdoor"])


def soft_event_phrase(label: str) -> str:
    mapping = {
        "Bird": "鳥のような鳴き声",
        "Bird vocalization, bird call, bird song": "鳥のような鳴き声",
        "Speech": "人の声のような発声",
        "Conversation": "複数人の話し声のような気配",
        "Animal": "生き物のような気配",
        "Meow": "小さく鋭い鳴き声のような音",
        "Cat": "小動物のような声",
        "Cry": "泣き声のような音",
        "Crying, sobbing": "泣き声のような音",
        "Baby cry, infant cry": "幼い発声のような音",
        "Vehicle": "機械音や移動体のような音",
        "Car": "車両のような走行音",
        "Engine": "エンジンのような低い機械音",
        "Wind": "風のような流れ音",
        "Wind noise (microphone)": "風のような流れ音",
        "Rain": "雨のような環境音",
        "Water": "水の流れのような音",
        "Ocean": "広い水辺を思わせるうねり音",
        "Waves, surf": "波が寄せる音",
        "Slosh": "水が跳ね返る音",
        "Waterfall": "強い水流のような音",
        "Footsteps": "足音のようなリズム",
        "Music": "音楽的な成分",
        "Silence": "静けさ",
    }
    return mapping.get(label, f"{label} のような音")


def english_prompt_seed(label: str) -> str:
    mapping = {
        "forest": "deep forest atmosphere",
        "field": "open grass field",
        "mountain": "mountain slope landscape",
        "riverside": "riverbank ambience",
        "roadside": "roadside atmosphere",
        "residential": "quiet residential edge",
        "urban": "urban edge ambience",
        "indoor": "interior ambience",
        "Bird": "bird call in background",
        "Speech": "distant human-like voice",
        "Animal": "animal presence",
        "Meow": "small sharp cry-like sound",
        "Cat": "small animal voice",
        "Vehicle": "distant vehicle sound",
        "Car": "passing car sound",
        "Engine": "low mechanical hum",
        "Wind": "soft wind layer",
        "Wind noise (microphone)": "wind texture",
        "Water": "water sound texture",
        "Ocean": "ocean ambience",
        "Waves, surf": "waves crashing",
        "Slosh": "water slosh",
        "Waterfall": "strong water flow",
        "Silence": "quiet atmosphere",
    }
    return mapping.get(label, label.lower().replace("_", " "))


# =========================================================
# Input extraction
# =========================================================
def extract_primary_space(space_judgement: Dict[str, Any]) -> str:
    attrs = space_judgement.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("primary_space")
        if value:
            return normalize_space_label(str(value))

    for key in ["primary_space", "top_space", "predicted_space"]:
        value = space_judgement.get(key)
        if value:
            return normalize_space_label(str(value))

    ranking = ensure_list(space_judgement.get("space_ranking"))
    if ranking:
        first = ranking[0]
        if isinstance(first, dict):
            return normalize_space_label(str(first.get("label") or first.get("space") or "unknown"))
        return normalize_space_label(str(first))
    return "unknown"


def extract_secondary_space(space_judgement: Dict[str, Any]) -> str:
    attrs = space_judgement.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("secondary_space")
        if value:
            return normalize_space_label(str(value))

    for key in ["secondary_space", "sub_space"]:
        value = space_judgement.get(key)
        if value:
            return normalize_space_label(str(value))

    ranking = ensure_list(space_judgement.get("space_ranking"))
    if len(ranking) >= 2:
        second = ranking[1]
        if isinstance(second, dict):
            return normalize_space_label(str(second.get("label") or second.get("space") or "unknown"))
        return normalize_space_label(str(second))
    return "unknown"


def extract_environment_family(space_judgement: Dict[str, Any], primary_space: str) -> str:
    fam = str(space_judgement.get("environment_family") or "").strip().lower()
    if fam:
        return fam
    if primary_space in ["forest", "field", "mountain", "riverside"]:
        return "nature"
    if primary_space in ["roadside", "urban", "residential"]:
        return "urban"
    if primary_space == "indoor":
        return "indoor"
    return "nature"

def extract_space_judgement_facts(space_judgement: Dict[str, Any]) -> Dict[str, Any]:
    final_label = str(
        space_judgement.get("final_label")
        or space_judgement.get("final_space_label")
        or space_judgement.get("label")
        or space_judgement.get("judged_space")
        or ""
    ).strip()

    confidence = space_judgement.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None

    reason = ensure_list(space_judgement.get("reason"))[:8]
    timeline = ensure_list(space_judgement.get("timeline"))[:8]

    attrs = space_judgement.get("attributes")
    is_mixed_or_ambiguous = False

    if final_label in {"mixed_ambiguous", "unknown"}:
        is_mixed_or_ambiguous = True

    if isinstance(attrs, dict):
        if attrs.get("mixed") is True or attrs.get("ambiguous") is True:
            is_mixed_or_ambiguous = True

    return {
        "final_label": final_label or "unknown",
        "confidence": confidence,
        "reason": [str(x).strip() for x in reason if str(x).strip()],
        "timeline": timeline,
        "is_mixed_or_ambiguous": is_mixed_or_ambiguous,
    }


def list_global_audio_events(audio_events: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["global_top_events", "global_audio_events", "global_events", "events_global"]:
        value = audio_events.get(key)
        if isinstance(value, list):
            return value
    return []


def list_segment_audio_events(audio_events: Dict[str, Any]) -> List[Dict[str, Any]]:
    # panns_v1 対応
    if isinstance(audio_events.get("segments"), list):
        rows = []
        for seg in audio_events["segments"]:
            if not isinstance(seg, dict):
                continue
            top_events = ensure_list(seg.get("top_events"))
            labels = []
            for ev in top_events:
                if isinstance(ev, dict):
                    lbl = str(ev.get("label") or "").strip()
                    if lbl:
                        labels.append(lbl)
            rows.append({
                "segment_id": seg.get("segment_id") or f"segment_{seg.get('segment_index', 'unknown')}",
                "raw_labels": labels,
            })
        return rows

    for key in ["segment_audio_events", "segment_events", "events_by_segment"]:
        value = audio_events.get(key)
        if isinstance(value, list):
            return value
    return []


def strong_global_event_labels(audio_events: Dict[str, Any]) -> List[str]:
    events = list_global_audio_events(audio_events)
    normalized = []

    excluded_labels = {
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Environment",
    }

    for ev in events:
        if not isinstance(ev, dict):
            continue

        label = str(ev.get("label") or ev.get("name") or "").strip()
        if not label or label in excluded_labels:
            continue

        try:
            mean_score = float(ev.get("mean_score", ev.get("score", ev.get("score_sum", 0.0))) or 0.0)
        except Exception:
            mean_score = 0.0

        try:
            hit_count = int(ev.get("hit_count", 1) or 1)
        except Exception:
            hit_count = 1

        normalized.append({
            "label": label,
            "mean_score": mean_score,
            "hit_count": hit_count,
        })

    if not normalized:
        return []

    normalized.sort(key=lambda x: x["mean_score"], reverse=True)
    top_score = normalized[0]["mean_score"]

    labels = []
    for ev in normalized[:8]:
        label = ev["label"]
        mean_score = ev["mean_score"]
        hit_count = ev["hit_count"]

        threshold = 0.14
        min_hits = 1

        if label in {"Speech", "Animal", "Vehicle"}:
            threshold = 0.16
        if label in {"Meow", "Cat", "Baby cry, infant cry", "Cry", "Crying, sobbing"}:
            threshold = 0.18
        if label in {"Waves, surf", "Ocean", "Slosh", "Waterfall", "Wind noise (microphone)", "Silence"}:
            threshold = 0.10

        relative_ok = mean_score >= top_score * 0.55
        absolute_ok = mean_score >= threshold and hit_count >= min_hits

        if absolute_ok or relative_ok:
            labels.append(label)

    return clamp_tags(labels, 8)


def extract_global_event_strengths(audio_events: Dict[str, Any]) -> Dict[str, float]:
    events = list_global_audio_events(audio_events)
    out: Dict[str, float] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("label") or ev.get("name") or "").strip()
        if not label:
            continue

        try:
            mean_score = float(ev.get("mean_score", ev.get("score", 0.0)) or 0.0)
        except Exception:
            mean_score = 0.0

        out[label] = mean_score

    return out


def classify_scene_context(
    primary_space: str,
    secondary_space: str,
    environment_family: str,
    event_strengths: Dict[str, float],
) -> Dict[str, Any]:
    speech = event_strengths.get("Speech", 0.0) + event_strengths.get("Conversation", 0.0)
    vehicle = (
        event_strengths.get("Vehicle", 0.0)
        + event_strengths.get("Car", 0.0)
        + event_strengths.get("Engine", 0.0)
        + event_strengths.get("Bus", 0.0)
    )
    music = event_strengths.get("Music", 0.0)
    footsteps = event_strengths.get("Footsteps", 0.0) + event_strengths.get("Shuffle", 0.0)

    water = (
        event_strengths.get("Ocean", 0.0)
        + event_strengths.get("Waves, surf", 0.0)
        + event_strengths.get("Water", 0.0)
        + event_strengths.get("Slosh", 0.0)
        + event_strengths.get("Waterfall", 0.0)
    )
    wind = event_strengths.get("Wind", 0.0) + event_strengths.get("Wind noise (microphone)", 0.0)

    animal = (
        event_strengths.get("Animal", 0.0)
        + event_strengths.get("Meow", 0.0)
        + event_strengths.get("Cat", 0.0)
        + event_strengths.get("Domestic animals, pets", 0.0)
    )

    urban_score = speech + vehicle + music + footsteps
    waterside_score = water + 0.35 * wind
    nature_score = 0.0

    if primary_space == "urban":
        urban_score += 0.35
    if secondary_space == "urban":
        urban_score += 0.15
    if environment_family in {"urban", "roadside"}:
        urban_score += 0.15

    if primary_space == "riverside":
        waterside_score += 0.20
    if secondary_space == "riverside":
        waterside_score += 0.10

    if primary_space in {"forest", "field", "mountain"}:
        nature_score += 0.30
    if secondary_space in {"forest", "field", "mountain"}:
        nature_score += 0.12
    if environment_family == "nature":
        nature_score += 0.12

    # 動物系は最優先
    if animal >= 0.42:
        if primary_space == "riverside" or secondary_space == "riverside":
            label = "animal_waterside_scene"
            reason = f"animal evidence dominant near waterside: animal={animal:.3f}, urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}"
        elif primary_space in {"forest", "field", "mountain"} or secondary_space in {"forest", "field", "mountain"}:
            label = "animal_nature_scene"
            reason = f"animal evidence dominant in natural scene: animal={animal:.3f}, urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}"
        elif primary_space == "indoor":
            label = "animal_indoor_scene"
            reason = f"animal evidence dominant in indoor-like scene: animal={animal:.3f}, urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}"
        else:
            label = "animal_quiet_scene"
            reason = f"animal evidence dominant in ambiguous scene: animal={animal:.3f}, urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}"
    elif urban_score >= waterside_score + 0.20 and urban_score >= nature_score + 0.15:
        label = "urban_activity"
        reason = f"urban evidence dominant: urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}, animal={animal:.3f}"
    elif waterside_score >= urban_score + 0.20 and waterside_score >= nature_score:
        label = "coastal_waterside"
        reason = f"waterside evidence dominant: urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}, animal={animal:.3f}"
    elif primary_space == "indoor":
        label = "indoor_scene"
        reason = f"ambiguous but primary_space=indoor: urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}, animal={animal:.3f}"
    else:
        label = "mixed_edge_scene"
        reason = f"close scores fallback: urban={urban_score:.3f}, waterside={waterside_score:.3f}, nature={nature_score:.3f}, animal={animal:.3f}"

    return {
        "resolved_scene_bias": label,
        "resolved_reason": reason,
        "urban_score": round(urban_score, 4),
        "waterside_score": round(waterside_score, 4),
        "nature_score": round(nature_score, 4),
        "animal_score": round(animal, 4),
        "event_strengths": {
            "speech": round(speech, 4),
            "vehicle": round(vehicle, 4),
            "music": round(music, 4),
            "footsteps": round(footsteps, 4),
            "water": round(water, 4),
            "wind": round(wind, 4),
            "animal": round(animal, 4),
        },
    }

def summarize_audio_events(audio_events: Dict[str, Any]) -> str:
    labels = strong_global_event_labels(audio_events)
    if not labels:
        return "目立つ音イベントの断定は難しいが、環境音の層が感じられる。"

    phrases = [soft_event_phrase(x) for x in labels[:4]]
    if len(phrases) == 1:
        return f"{phrases[0]} が含まれている。"
    return "、".join(phrases[:-1]) + "、" + phrases[-1] + " が重なっている。"


def build_segment_audio_hints(audio_events: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list_segment_audio_events(audio_events)
    result: List[Dict[str, Any]] = []

    for row in rows[:8]:
        if not isinstance(row, dict):
            continue

        segment_id = str(row.get("segment_id") or row.get("id") or "")
        labels = ensure_list(row.get("raw_labels") or row.get("labels") or row.get("top_labels"))
        labels = [str(x).strip() for x in labels if str(x).strip()]
        labels = labels[:4]

        phrases = [soft_event_phrase(x) for x in labels[:3]]
        if phrases:
            desc = "、".join(phrases) + " が感じられる"
        else:
            desc = "顕著な音イベントの断定は難しい"

        result.append({
            "segment_id": segment_id or "segment_unknown",
            "description": desc,
            "raw_labels": labels,
        })

    return result


def build_timeline_summary(space_similarity: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not space_similarity:
        return []

    rows = ensure_list(space_similarity.get("segment_space_scores"))
    result: List[Dict[str, Any]] = []

    for row in rows[:8]:
        if not isinstance(row, dict):
            continue

        segment_id = str(row.get("segment_id") or "segment_unknown")
        top_space = row.get("top_space") or row.get("best_space")
        top_score = row.get("top_score") or row.get("best_score")

        if not top_space:
            candidates = ensure_list(row.get("scores"))
            if candidates and isinstance(candidates[0], dict):
                top_space = candidates[0].get("label") or candidates[0].get("space_id") or candidates[0].get("space")
                top_score = candidates[0].get("final_score") or candidates[0].get("score")

        top_space_norm = normalize_space_label(str(top_space or "unknown"))
        score_text = ""
        if top_score is not None:
            try:
                score_text = f"（{float(top_score):.4f}）"
            except Exception:
                score_text = ""

        if top_space_norm == "forest":
            desc = f"森のような奥行きや反射感が優勢 {score_text}".strip()
        elif top_space_norm == "roadside":
            desc = f"開けた人工環境のような気配が優勢 {score_text}".strip()
        elif top_space_norm == "riverside":
            desc = f"水辺のような開放感が優勢 {score_text}".strip()
        elif top_space_norm == "field":
            desc = f"草地のような開けた気配が優勢 {score_text}".strip()
        elif top_space_norm == "mountain":
            desc = f"山あいのような広がりが優勢 {score_text}".strip()
        else:
            desc = f"{space_to_ja(top_space_norm)} 傾向が優勢 {score_text}".strip()

        result.append({
            "segment_id": segment_id,
            "description": desc,
        })

    return result

# =========================================================
# Evidence control / postcheck
# =========================================================
def derive_forbidden_terms(payload: Dict[str, Any]) -> List[str]:
    event_tags = set(str(x).strip() for x in ensure_list(payload.get("audio_event_tags")) if str(x).strip())
    strengths = payload.get("audio_event_strengths") or {}

    forbidden: List[str] = []

    bird_labels = {
        "Bird",
        "Bird vocalization, bird call, bird song",
    }
    if not (event_tags & bird_labels):
        forbidden.extend(["鳥", "さえずり", "羽ばたき", "カモメ", "群鳥"])

    speech_score = float(strengths.get("Speech", 0.0) + strengths.get("Conversation", 0.0))
    footsteps_score = float(strengths.get("Footsteps", 0.0))

    if speech_score < 0.18 and footsteps_score < 0.12:
        forbidden.extend(["人物", "人影", "男", "女", "子ども", "誰かが立つ", "佇む"])

    if "Rain" not in event_tags:
        forbidden.extend(["雨", "雨粒", "濡れた路地", "雨上がり"])

    return clamp_tags(forbidden, 32)


def build_scene_bias(payload: Dict[str, Any]) -> Dict[str, Any]:
    primary_space = normalize_space_label(payload.get("primary_space"))
    secondary_space = normalize_space_label(payload.get("secondary_space"))
    environment_family = str(payload.get("environment_family") or "").strip().lower()

    event_strengths = payload.get("audio_event_strengths")
    if not isinstance(event_strengths, dict):
        event_strengths = {}

    return classify_scene_context(
        primary_space=primary_space,
        secondary_space=secondary_space,
        environment_family=environment_family,
        event_strengths=event_strengths,
    )


def remove_forbidden_terms_from_text(text: str, forbidden_terms: List[str]) -> str:
    if not isinstance(text, str):
        return text

    cleaned = text
    for term in forbidden_terms:
        if not term:
            continue
        cleaned = cleaned.replace(term, "")

    cleaned = re.sub(r"、、+", "、", cleaned)
    cleaned = re.sub(r"。。+", "。", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^、+", "", cleaned)
    cleaned = re.sub(r"^。+", "", cleaned)
    return cleaned.strip()


def postcheck_scene(scene: Dict[str, Any], payload: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    forbidden_terms = derive_forbidden_terms(payload)
    removed_terms: List[str] = []

    checked = dict(scene)

    # 文字列フィールドの軽い禁止語除去だけ残す
    for key in ["scene_title", "scene_summary", "audio_event_summary", "narrative_hook"]:
        value = checked.get(key)
        if isinstance(value, str):
            original = value
            cleaned = remove_forbidden_terms_from_text(value, forbidden_terms)
            checked[key] = cleaned if cleaned else base.get(key, original)
            if cleaned != original:
                for term in forbidden_terms:
                    if term in original:
                        removed_terms.append(term)

    # visual_hints の軽い禁止語除去
    visual_hints = []
    for hint in ensure_list(checked.get("visual_hints")):
        h = str(hint).strip()
        if not h:
            continue
        cleaned = remove_forbidden_terms_from_text(h, forbidden_terms)
        if cleaned:
            visual_hints.append(cleaned)
            if cleaned != h:
                for term in forbidden_terms:
                    if term in h:
                        removed_terms.append(term)

    if len(visual_hints) < 3:
        visual_hints = ensure_list(base.get("visual_hints"))[:8]

    checked["visual_hints"] = clamp_tags([str(x) for x in visual_hints], 8)

    # prompt_seed_en は最小限の整形だけ
    bridge = checked.get("manga_prompt_bridge")
    if isinstance(bridge, dict):
        prompt_seed_en = [
            str(x).strip()
            for x in ensure_list(bridge.get("prompt_seed_en"))
            if str(x).strip()
        ]
        bridge["prompt_seed_en"] = clamp_tags(prompt_seed_en, 16)
        checked["manga_prompt_bridge"] = bridge

    checked["_postcheck"] = {
        "applied": True,
        "mode": "light",
        "forbidden_terms": forbidden_terms,
        "removed_terms": clamp_tags(removed_terms, 20),
    }

    return checked

def build_scene_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    primary_space = normalize_space_label(payload.get("primary_space"))
    secondary_space = normalize_space_label(payload.get("secondary_space"))

    sj = payload.get("space_judgement_summary") or {}
    final_label = str(sj.get("final_label") or "unknown")
    confidence = sj.get("confidence")
    is_mixed = bool(sj.get("is_mixed_or_ambiguous"))

    event_strengths = payload.get("audio_event_strengths") or {}

    speech = float(event_strengths.get("Speech", 0.0) + event_strengths.get("Conversation", 0.0))
    vehicle = float(
        event_strengths.get("Vehicle", 0.0)
        + event_strengths.get("Car", 0.0)
        + event_strengths.get("Engine", 0.0)
        + event_strengths.get("Bus", 0.0)
    )
    water = float(
        event_strengths.get("Ocean", 0.0)
        + event_strengths.get("Waves, surf", 0.0)
        + event_strengths.get("Water", 0.0)
        + event_strengths.get("Slosh", 0.0)
        + event_strengths.get("Waterfall", 0.0)
    )
    wind = float(event_strengths.get("Wind", 0.0) + event_strengths.get("Wind noise (microphone)", 0.0))
    animal = float(
        event_strengths.get("Animal", 0.0)
        + event_strengths.get("Meow", 0.0)
        + event_strengths.get("Cat", 0.0)
        + event_strengths.get("Domestic animals, pets", 0.0)
    )

    candidates: List[Dict[str, Any]] = [
        {
            "candidate_id": "A",
            "label": "space_judgement_root",
            "priority": 1,
            "focus": "06_space_judgement を主根拠にする",
            "hint": f"final_label={final_label}, primary_space={primary_space}, secondary_space={secondary_space}, confidence={confidence}",
        }
    ]

    material_flags = []
    if animal >= 0.18:
        material_flags.append("animal_like")
    if water >= 0.18:
        material_flags.append("water_like")
    if wind >= 0.16:
        material_flags.append("wind_like")
    if speech >= 0.18:
        material_flags.append("speech_like")
    if vehicle >= 0.18:
        material_flags.append("vehicle_like")

    if material_flags:
        candidates.append({
            "candidate_id": "B",
            "label": "audio_material_check",
            "priority": 2,
            "focus": "04-05 の素材を情景材料として使う",
            "hint": ", ".join(material_flags),
        })

    if is_mixed or secondary_space not in {"", "unknown", primary_space}:
        candidates.append({
            "candidate_id": "C",
            "label": "mixed_or_ambiguous",
            "priority": 3,
            "focus": "曖昧性を保持する",
            "hint": "単一空間へ無理に断定せず、曖昧な屋外/屋内表現を許容する",
        })

    return candidates[:3]

# =========================================================
# Prompt builders
# =========================================================
def build_user_prompt(payload: Dict[str, Any]) -> str:
    facts = {
        "primary_space": payload.get("primary_space"),
        "secondary_space": payload.get("secondary_space"),
        "environment_family": payload.get("environment_family"),
        "space_judgement_summary": payload.get("space_judgement_summary"),
        "audio_event_tags": ensure_list(payload.get("audio_event_tags"))[:5],
        "audio_event_summary": payload.get("audio_event_summary"),
        "segment_audio_hints": ensure_list(payload.get("segment_audio_hints"))[:8],
        "audio_event_strengths": payload.get("audio_event_strengths"),
    }

    return (
        "以下の情報だけを使って、情景の核となる最小JSONを作成してください。\n"
        "曖昧な場合は断定を弱めてください。\n"
        "人物・動物・雨・都市要素は根拠が弱ければ確定しないでください。\n\n"
        f"facts:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        "出力フォーマットは必ず次の形です。\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "scene_title": "短い日本語タイトル",\n'
        '  "scene_summary": "2〜4文の日本語説明",\n'
        '  "mood_tags": ["日本語短語を4〜8個"],\n'
        '  "environment_tags": ["空間ラベルを2〜4個"],\n'
        '  "visual_hints": ["画像生成に使える具体的な日本語ヒントを5〜8個"],\n'
        '  "narrative_hook": "1〜2文の日本語",\n'
        '  "subject_hint": {\n'
        '    "subject_type": "human | animal | none",\n'
        '    "subject_role": "短い日本語",\n'
        '    "appearance_hint": "日本語で1文",\n'
        '    "framing_hint": "wide | distant | side_view | back_view | small_in_frame",\n'
        '    "confidence": 0.0,\n'
        '    "reason": "日本語で1文"\n'
        "  }\n"
        "}\n"
        "END_JSON"
    )

# =========================================================
# Local LLM
# =========================================================
def load_local_llm() -> None:
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    if not LOCAL_LLM_ENABLED:
        raise RuntimeError("local llm disabled by SWALLOW_LOCAL_ENABLED")

    print(f"[scene_interpreter] loading local LLM on CPU: {LOCAL_MODEL_NAME}")

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
        model_kwargs["token"] = HF_TOKEN

    _tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_NAME,
        **tokenizer_kwargs,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        **model_kwargs,
    )

    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model.eval()
    try:
        _model.to("cpu")
    except Exception:
        pass

    print("[scene_interpreter] local LLM loaded on CPU")


def extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM response is empty")

    text = text.strip()

    match = re.search(r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain BEGIN_JSON/END_JSON block")

    json_text = match.group(1).strip()
    obj = json.loads(json_text)

    if not isinstance(obj, dict):
        raise ValueError("LLM JSON must be dict")

    return obj

REQUIRED_SCENE_KEYS = [
    "scene_title",
    "scene_summary",
    "mood_tags",
    "environment_tags",
    "timeline_summary",
    "visual_hints",
    "audio_event_tags",
    "audio_event_summary",
    "segment_audio_hints",
    "narrative_hook",
    "manga_prompt_bridge",
    "subject_hint",
]

MIN_SCENE_KEYS = [
    "scene_title",
    "scene_summary",
    "mood_tags",
    "environment_tags",
    "visual_hints",
    "narrative_hook",
    "subject_hint",
]

def validate_scene_json(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("scene JSON must be dict")

    missing = [k for k in REQUIRED_SCENE_KEYS if k not in obj]
    if missing:
        raise ValueError(f"scene JSON missing required keys: {missing}")

    if not isinstance(obj.get("mood_tags"), list):
        raise ValueError("mood_tags must be list")

    if not isinstance(obj.get("environment_tags"), list):
        raise ValueError("environment_tags must be list")

    if not isinstance(obj.get("visual_hints"), list):
        raise ValueError("visual_hints must be list")

    if not isinstance(obj.get("audio_event_tags"), list):
        raise ValueError("audio_event_tags must be list")

    if not isinstance(obj.get("segment_audio_hints"), list):
        raise ValueError("segment_audio_hints must be list")

    if not isinstance(obj.get("timeline_summary"), list):
        raise ValueError("timeline_summary must be list")

    bridge = obj.get("manga_prompt_bridge")
    if not isinstance(bridge, dict):
        raise ValueError("manga_prompt_bridge must be dict")

    subject_hint = obj.get("subject_hint")
    if not isinstance(subject_hint, dict):
        raise ValueError("subject_hint must be dict")

    required_subject_keys = [
        "subject_type",
        "subject_role",
        "appearance_hint",
        "framing_hint",
        "confidence",
        "reason",
    ]
    missing_subject = [k for k in required_subject_keys if k not in subject_hint]
    if missing_subject:
        raise ValueError(f"subject_hint missing required keys: {missing_subject}")

    if str(subject_hint.get("subject_type", "")).strip() not in {"human", "animal", "none"}:
        raise ValueError("subject_hint.subject_type must be one of: human, animal, none")

    if not isinstance(subject_hint.get("subject_role"), str):
        raise ValueError("subject_hint.subject_role must be str")

    if not isinstance(subject_hint.get("appearance_hint"), str):
        raise ValueError("subject_hint.appearance_hint must be str")

    if not isinstance(subject_hint.get("framing_hint"), str):
        raise ValueError("subject_hint.framing_hint must be str")

    if not isinstance(subject_hint.get("reason"), str):
        raise ValueError("subject_hint.reason must be str")

    try:
        float(subject_hint.get("confidence"))
    except Exception:
        raise ValueError("subject_hint.confidence must be float-convertible")

def validate_min_scene_json(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("min scene JSON must be dict")

    missing = [k for k in MIN_SCENE_KEYS if k not in obj]
    if missing:
        raise ValueError(f"min scene JSON missing required keys: {missing}")

    if not isinstance(obj.get("scene_title"), str):
        raise ValueError("scene_title must be str")

    if not isinstance(obj.get("scene_summary"), str):
        raise ValueError("scene_summary must be str")

    if not isinstance(obj.get("mood_tags"), list):
        raise ValueError("mood_tags must be list")

    if not isinstance(obj.get("environment_tags"), list):
        raise ValueError("environment_tags must be list")

    if not isinstance(obj.get("visual_hints"), list):
        raise ValueError("visual_hints must be list")

    if not isinstance(obj.get("narrative_hook"), str):
        raise ValueError("narrative_hook must be str")

    subject_hint = obj.get("subject_hint")
    if not isinstance(subject_hint, dict):
        raise ValueError("subject_hint must be dict")

    required_subject_keys = [
        "subject_type",
        "subject_role",
        "appearance_hint",
        "framing_hint",
        "confidence",
        "reason",
    ]
    missing_subject = [k for k in required_subject_keys if k not in subject_hint]
    if missing_subject:
        raise ValueError(f"subject_hint missing required keys: {missing_subject}")

    if str(subject_hint.get("subject_type", "")).strip() not in {"human", "animal", "none"}:
        raise ValueError("subject_hint.subject_type must be one of: human, animal, none")

    try:
        float(subject_hint.get("confidence"))
    except Exception:
        raise ValueError("subject_hint.confidence must be float-convertible")    

def call_local_swallow_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    load_local_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(_tokenizer, "apply_chat_template"):
        prompt_text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = (
            f"<|system|>\n{system_prompt}\n\n"
            f"<|user|>\n{user_prompt}\n\n"
            f"<|assistant|>\n"
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    inputs = None
    output_ids = None
    generated_ids = None
    text = None

    try:
        if use_cuda:
            _model.to(device=device, dtype=torch.float16)
            _model.eval()
            torch.cuda.empty_cache()

        inputs = _tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=min(LOCAL_LLM_MAX_NEW_TOKENS, 450),
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                use_cache=False,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 成功/失敗に関係なく raw を保存
        try:
            debug_dir = RESULT_ROOT / "_debug_scene_interpreter"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = str(int(time.time()))
            with open(debug_dir / f"scene_llm_raw_{ts}.txt", "w", encoding="utf-8") as f:
                f.write(text or "")
        except Exception:
            pass

        parsed = extract_json_object(text)
        validate_min_scene_json(parsed)
        return parsed

    except Exception as e:
        preview = text[:1200] if isinstance(text, str) else ""
        raise ValueError(f"failed to parse/validate LLM JSON: {e}; raw_preview={preview!r}") from e

    finally:
        try:
            del inputs
        except Exception:
            pass
        try:
            del output_ids
        except Exception:
            pass
        try:
            del generated_ids
        except Exception:
            pass

        if use_cuda:
            try:
                _model.to("cpu")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

# =========================================================
# Base scene (schema anchor, not fallback output)
# =========================================================
def build_base_scene(payload: Dict[str, Any]) -> Dict[str, Any]:
    primary_space = normalize_space_label(payload.get("primary_space"))
    secondary_space = normalize_space_label(payload.get("secondary_space"))
    environment_family = str(payload.get("environment_family") or "").strip().lower()

    event_tags = [str(x).strip() for x in ensure_list(payload.get("audio_event_tags")) if str(x).strip()]
    audio_event_summary = str(payload.get("audio_event_summary") or "").strip()

    sj = payload.get("space_judgement_summary") or {}
    confidence = sj.get("confidence")
    is_mixed = bool(sj.get("is_mixed_or_ambiguous"))

    environment_tags: List[str] = []
    environment_tags.extend(environment_family_to_tags(environment_family))
    if primary_space != "unknown":
        environment_tags.append(primary_space)
    if secondary_space not in {"unknown", "", primary_space}:
        environment_tags.append(secondary_space)
    environment_tags = clamp_tags(environment_tags, 4)

    mood_tags = ["観察的", "静けさ", "環境音", "余韻"]
    if any(x in event_tags for x in ["Water", "Ocean", "Waves, surf", "Slosh", "Waterfall"]):
        mood_tags.append("湿度感")
    if any(x in event_tags for x in ["Wind", "Wind noise (microphone)"]):
        mood_tags.append("風感")
    if any(x in event_tags for x in ["Speech", "Conversation", "Vehicle", "Car", "Engine"]):
        mood_tags.append("生活感")
    mood_tags = clamp_tags(mood_tags, 8)

    if is_mixed or primary_space == "unknown":
        scene_title = "曖昧な環境の情景"
        scene_summary = "空間は一つに断定しきれないが、複数の環境音が重なっている。音素材をもとに、無理のない範囲で情景としてまとめる。"
        visual_hints = ["曖昧な背景", "重なる環境音", "開けすぎない構図"]
        prompt_seed_en = ["ambiguous environment", "layered ambience", "uncertain space"]
    elif primary_space == "forest":
        scene_title = "森の気配がある情景"
        scene_summary = "06_space_judgement では自然寄りの空間が主とされている。その前提を保ちながら、音素材から無理のない範囲で情景を組み立てる。"
        visual_hints = ["木立", "自然光", "奥行き"]
        prompt_seed_en = ["forest", "outdoor", "natural ambience"]
    elif primary_space == "riverside":
        scene_title = "水辺の情景"
        scene_summary = "06_space_judgement では水辺寄りの空間が主とされている。その前提を保ちながら、水や風の気配を中心に情景を組み立てる。"
        visual_hints = ["水面", "風", "開けた景色"]
        prompt_seed_en = ["riverside", "water", "outdoor"]
    elif primary_space in {"urban", "roadside", "residential"}:
        scene_title = "人工環境寄りの情景"
        scene_summary = "06_space_judgement では人工環境寄りの空間が主とされている。ただし音素材と矛盾が強い場合は断定を弱め、自然な範囲でまとめる。"
        visual_hints = ["建物", "道路", "生活感"]
        prompt_seed_en = ["urban", "street", "city ambience"]
    elif primary_space == "indoor":
        scene_title = "屋内の情景"
        scene_summary = "06_space_judgement では屋内寄りの空間が主とされている。その前提を保ちながら、近距離の音や反射感を中心に情景をまとめる。"
        visual_hints = ["室内", "壁", "近い音"]
        prompt_seed_en = ["indoor", "room", "interior"]
    else:
        scene_title = f"{space_to_ja(primary_space)}の情景"
        scene_summary = "06_space_judgement の結果を主根拠にしつつ、音素材から無理のない情景をまとめる。"
        visual_hints = [space_to_ja(primary_space), "環境音", "空間の広がり"]
        prompt_seed_en = [english_prompt_seed(primary_space), "ambient scene", "sound-based interpretation"]

    if confidence is not None and confidence < 0.58:
        scene_summary += " 空間確信度は高くないため、断定はやや弱めに保つ。"

    narrative_hook = "空間判定を土台にしつつ、音素材から自然に見える一場面へ整える。"

    animagine_direction_ja = clamp_tags([
        "空間の前提を崩さない",
        "音素材と矛盾する要素を足さない",
        "背景を主役にしすぎない",
        "視覚化しやすい情景にする",
        "断定しすぎない",
        "過剰な演出を避ける",
    ], 6)

    return {
        "scene_title": scene_title,
        "scene_summary": scene_summary,
        "mood_tags": mood_tags,
        "environment_tags": environment_tags,
        "timeline_summary": ensure_list(payload.get("timeline_summary"))[:8],
        "visual_hints": clamp_tags(visual_hints, 8),
        "audio_event_tags": event_tags[:8],
        "audio_event_summary": audio_event_summary,
        "segment_audio_hints": ensure_list(payload.get("segment_audio_hints"))[:8],
        "narrative_hook": narrative_hook,
        "manga_prompt_bridge": {
            "scene_core_ja": scene_summary,
            "story_hook_ja": narrative_hook,
            "animagine_direction_ja": animagine_direction_ja,
            "prompt_seed_en": clamp_tags(prompt_seed_en, 16),
        },
        "subject_hint": {
            "subject_type": "none",
            "subject_role": "情景主導",
            "appearance_hint": "主役を明確に置かず、風景や空気感を主軸にする",
            "framing_hint": "wide",
            "confidence": 0.35,
            "reason": "音だけでは主役を強く断定しないため"
        },
    }

# =========================================================
# Merge / schema ensure
# =========================================================
def ensure_schema(scene: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)

    for key in ["scene_title", "scene_summary", "audio_event_summary", "narrative_hook"]:
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()

    for key, max_items in [
        ("mood_tags", 8),
        ("environment_tags", 4),
        ("visual_hints", 8),
        ("audio_event_tags", 8),
    ]:
        value = scene.get(key)
        if isinstance(value, list):
            result[key] = clamp_tags([str(x) for x in value], max_items)

    timeline = scene.get("timeline_summary")
    if isinstance(timeline, list) and timeline:
        clean_timeline = []
        for row in timeline[:8]:
            if not isinstance(row, dict):
                continue
            clean_timeline.append({
                "segment_id": str(row.get("segment_id") or "segment_unknown"),
                "description": str(row.get("description") or "").strip() or "音の変化がある",
            })
        if clean_timeline:
            result["timeline_summary"] = clean_timeline

    hints = scene.get("segment_audio_hints")
    if isinstance(hints, list) and hints:
        clean_hints = []
        for row in hints[:8]:
            if not isinstance(row, dict):
                continue
            clean_hints.append({
                "segment_id": str(row.get("segment_id") or "segment_unknown"),
                "description": str(row.get("description") or "").strip() or "音の気配がある",
                "raw_labels": [str(x) for x in ensure_list(row.get("raw_labels"))[:6]],
            })
        if clean_hints:
            result["segment_audio_hints"] = clean_hints

    bridge = scene.get("manga_prompt_bridge")
    if isinstance(bridge, dict):
        base_bridge = base.get("manga_prompt_bridge", {})
        merged_bridge = dict(base_bridge)

        for key in ["scene_core_ja", "story_hook_ja"]:
            value = bridge.get(key)
            if isinstance(value, str) and value.strip():
                merged_bridge[key] = value.strip()

        for key, max_items in [
            ("animagine_direction_ja", 6),
            ("prompt_seed_en", 16),
        ]:
            value = bridge.get(key)
            if isinstance(value, list):
                merged_bridge[key] = clamp_tags([str(x) for x in value], max_items)

        result["manga_prompt_bridge"] = merged_bridge
    subject_hint = scene.get("subject_hint")
    if isinstance(subject_hint, dict):
        base_subject = base.get("subject_hint", {})
        merged_subject = dict(base_subject)

        for key in ["subject_type", "subject_role", "appearance_hint", "framing_hint", "reason"]:
            value = subject_hint.get(key)
            if isinstance(value, str) and value.strip():
                merged_subject[key] = value.strip()

        conf = subject_hint.get("confidence")
        try:
            merged_subject["confidence"] = float(conf)
        except Exception:
            pass

        result["subject_hint"] = merged_subject

    return result

def build_prompt_seed_en_from_scene(scene: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    seeds: List[str] = []

    primary_space = normalize_space_label(payload.get("primary_space"))
    secondary_space = normalize_space_label(payload.get("secondary_space"))

    if primary_space and primary_space != "unknown":
        seeds.append(english_prompt_seed(primary_space))
    if secondary_space and secondary_space not in {"", "unknown", primary_space}:
        seeds.append(english_prompt_seed(secondary_space))

    for label in ensure_list(payload.get("audio_event_tags"))[:6]:
        s = str(label).strip()
        if s:
            seeds.append(english_prompt_seed(s))

    subject_hint = scene.get("subject_hint") or {}
    subject_type = str(subject_hint.get("subject_type") or "").strip()
    if subject_type == "human":
        seeds.extend(["human figure", "small subject", "natural pose"])
    elif subject_type == "animal":
        seeds.extend(["animal presence", "small animal", "subtle subject"])
    else:
        seeds.extend(["ambient scene", "no clear subject", "environment-led composition"])

    for hint in ensure_list(scene.get("visual_hints"))[:4]:
        s = str(hint).strip()
        if s:
            seeds.append(s.lower())

    return clamp_tags(seeds, 16)

def inflate_min_scene_to_full_scene(min_scene: Dict[str, Any], payload: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    full_scene = dict(base)

    for key in ["scene_title", "scene_summary", "narrative_hook"]:
        value = min_scene.get(key)
        if isinstance(value, str) and value.strip():
            full_scene[key] = value.strip()

    for key, max_items in [
        ("mood_tags", 8),
        ("environment_tags", 4),
        ("visual_hints", 8),
    ]:
        value = min_scene.get(key)
        if isinstance(value, list):
            full_scene[key] = clamp_tags([str(x) for x in value], max_items)

    subject_hint = min_scene.get("subject_hint")
    if isinstance(subject_hint, dict):
        base_subject = base.get("subject_hint", {})
        merged_subject = dict(base_subject)

        for key in ["subject_type", "subject_role", "appearance_hint", "framing_hint", "reason"]:
            value = subject_hint.get(key)
            if isinstance(value, str) and value.strip():
                merged_subject[key] = value.strip()

        try:
            merged_subject["confidence"] = float(subject_hint.get("confidence"))
        except Exception:
            pass

        full_scene["subject_hint"] = merged_subject

    # ここから先は Python 側で埋める
    full_scene["timeline_summary"] = ensure_list(payload.get("timeline_summary"))[:8]
    full_scene["audio_event_tags"] = [str(x).strip() for x in ensure_list(payload.get("audio_event_tags")) if str(x).strip()][:8]
    full_scene["audio_event_summary"] = str(payload.get("audio_event_summary") or "").strip()
    full_scene["segment_audio_hints"] = ensure_list(payload.get("segment_audio_hints"))[:8]

    full_scene["manga_prompt_bridge"] = {
        "scene_core_ja": full_scene["scene_summary"],
        "story_hook_ja": full_scene["narrative_hook"],
        "animagine_direction_ja": clamp_tags([
            "空間の前提を崩さない",
            "音素材と矛盾する要素を足さない",
            "背景を主役にしすぎない",
            "視覚化しやすい情景にする",
            "断定しすぎない",
            "過剰な演出を避ける",
        ], 6),
        "prompt_seed_en": build_prompt_seed_en_from_scene(full_scene, payload),
    }

    return full_scene

# =========================================================
# Payload builder
# =========================================================
def build_payload(
    features: Optional[Dict[str, Any]],
    audio_events: Dict[str, Any],
    space_judgement: Dict[str, Any],
    space_similarity: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    primary_space = extract_primary_space(space_judgement)
    secondary_space = extract_secondary_space(space_judgement)
    environment_family = extract_environment_family(space_judgement, primary_space)
    space_judgement_summary = extract_space_judgement_facts(space_judgement)

    event_strengths = extract_global_event_strengths(audio_events)
    strong_events = strong_global_event_labels(audio_events)

    payload = {
        "primary_space": primary_space,
        "secondary_space": secondary_space,
        "environment_family": environment_family,
        "space_judgement_summary": space_judgement_summary,
        "audio_event_tags": strong_events,
        "audio_event_strengths": event_strengths,
        "audio_event_summary": summarize_audio_events(audio_events),
        "timeline_summary": build_timeline_summary(space_similarity),
        "segment_audio_hints": build_segment_audio_hints(audio_events),
        "space_analysis": {
            "primary_space": primary_space,
            "secondary_space": secondary_space,
            "environment_family": environment_family,
        },
        "audio_features_summary": {},
    }

    if isinstance(features, dict):
        feature_source = features["features"] if "features" in features and isinstance(features["features"], dict) else features
        feature_keys = [
            "duration_sec",
            "rms_mean",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "zero_crossing_rate_mean",
            "silence_ratio",
        ]
        for k in feature_keys:
            if k in feature_source:
                payload["audio_features_summary"][k] = feature_source.get(k)

    return payload

# =========================================================
# Generation
# =========================================================
def generate_scene(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not LOCAL_LLM_ENABLED:
        raise RuntimeError("scene_interpreter requires local LLM, but SWALLOW_LOCAL_ENABLED is disabled")

    base = build_base_scene(payload)

    started = time.time()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)

    print("[scene_interpreter] trying local llm...")

    try:
        llm_scene_min = call_local_swallow_json(system_prompt, user_prompt)
        latency = round(time.time() - started, 3)

        inflated = inflate_min_scene_to_full_scene(llm_scene_min, payload, base)
        inflated = ensure_schema(inflated, base)
        inflated = postcheck_scene(inflated, payload, base)

        refined = refine_scene_with_self_check(inflated, payload, base)

        refined["_debug_llm"] = {
            "enabled": True,
            "used": True,
            "status": "success",
            "reason": "local llm min-json succeeded, inflated, postchecked, and self-reviewed",
            "latency_sec": latency,
        }

        print(f"[scene_interpreter] local LLM success ({latency}s)")
        return refined

    except Exception as e:
        latency = round(time.time() - started, 3)

        fallback = postcheck_scene(base, payload, base)

        fallback["_debug_llm"] = {
            "enabled": True,
            "used": False,
            "status": "failed",
            "reason": f"fallback to base scene: {e}",
            "latency_sec": latency,
        }
        print(f"[scene_interpreter] local LLM failed, fallback to base scene ({latency}s): {e}")
        return fallback
    
def build_system_prompt() -> str:
    return (
        "あなたはJSONのみを出力するAIです。\n"
        "説明文は禁止です。\n"
        "前置きは禁止です。\n"
        "後書きは禁止です。\n"
        "必ず有効なJSONオブジェクトを1つだけ返してください。\n\n"
        "厳守事項:\n"
        "- 出力は必ず BEGIN_JSON で開始し END_JSON で終了すること\n"
        "- BEGIN_JSON と END_JSON の間には JSON オブジェクトのみを書くこと\n"
        "- JSONの外に一切何も書かないこと\n"
        "- コメント禁止\n"
        "- 末尾カンマ禁止\n"
        "- キー欠落禁止\n"
        "- 不明な場合でも空文字、空配列、none などで埋めること\n"
    )

def build_self_check_system_prompt() -> str:
    return (
        "あなたはJSONのみを出力するレビューAIです。\n"
        "前置き禁止、説明禁止、後書き禁止です。\n"
        "必ず BEGIN_JSON と END_JSON の間に JSON オブジェクト1つだけを書いてください。\n"
        "大きく書き換えず、必要最小限の修正だけ行ってください。\n"
        "不明なものは無理に追加せず、既存値を維持してください。\n"
    )


def build_self_check_user_prompt(scene: Dict[str, Any], payload: Dict[str, Any]) -> str:
    facts = {
        "primary_space": payload.get("primary_space"),
        "secondary_space": payload.get("secondary_space"),
        "environment_family": payload.get("environment_family"),
        "audio_event_tags": ensure_list(payload.get("audio_event_tags"))[:5],
        "audio_event_summary": payload.get("audio_event_summary"),
        "segment_audio_hints": ensure_list(payload.get("segment_audio_hints"))[:8],
        "audio_event_strengths": payload.get("audio_event_strengths"),
    }

    compact_scene = {
        "scene_title": scene.get("scene_title"),
        "scene_summary": scene.get("scene_summary"),
        "mood_tags": scene.get("mood_tags"),
        "environment_tags": scene.get("environment_tags"),
        "visual_hints": scene.get("visual_hints"),
        "narrative_hook": scene.get("narrative_hook"),
        "subject_hint": scene.get("subject_hint"),
    }

    return (
        "以下の最小JSONをレビューしてください。\n"
        "修正が不要なら、そのまま返してください。\n"
        "修正は必要最小限だけ行ってください。\n\n"
        f"facts:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        f"current_scene:\n{json.dumps(compact_scene, ensure_ascii=False, indent=2)}\n\n"
        "出力形式:\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "scene_title": "短い日本語タイトル",\n'
        '  "scene_summary": "2〜4文の日本語説明",\n'
        '  "mood_tags": ["日本語短語を4〜8個"],\n'
        '  "environment_tags": ["空間ラベルを2〜4個"],\n'
        '  "visual_hints": ["画像生成に使える具体的な日本語ヒントを5〜8個"],\n'
        '  "narrative_hook": "1〜2文の日本語",\n'
        '  "subject_hint": {\n'
        '    "subject_type": "human | animal | none",\n'
        '    "subject_role": "短い日本語",\n'
        '    "appearance_hint": "日本語で1文",\n'
        '    "framing_hint": "wide | distant | side_view | back_view | small_in_frame",\n'
        '    "confidence": 0.0,\n'
        '    "reason": "日本語で1文"\n'
        "  }\n"
        "}\n"
        "END_JSON"
    )


def refine_scene_with_self_check(scene: Dict[str, Any], payload: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    try:
        system_prompt = build_self_check_system_prompt()
        user_prompt = build_self_check_user_prompt(scene, payload)

        reviewed_min = call_local_swallow_json(system_prompt, user_prompt)
        reviewed_full = inflate_min_scene_to_full_scene(reviewed_min, payload, base)
        reviewed_full = ensure_schema(reviewed_full, base)
        reviewed_full = postcheck_scene(reviewed_full, payload, base)
        return reviewed_full
    except Exception as e:
        print(f"[scene_interpreter] self-check skipped: {e}")
        return scene

# =========================================================
# Runner
# =========================================================
def run(audio_id: str) -> Dict[str, Any]:
    try:
        result_dir = RESULT_ROOT / audio_id
        result_dir.mkdir(parents=True, exist_ok=True)

        features_path = result_dir / "04_features.json"
        audio_events_path = result_dir / "05_audio_events.json"
        space_similarity_path = result_dir / "05_space_similarity.json"
        space_judgement_path = result_dir / "06_space_judgement.json"
        output_path = result_dir / OUTPUT_FILENAME

        if not audio_events_path.exists():
            raise FileNotFoundError(f"JSON not found: {audio_events_path}")
        if not space_judgement_path.exists():
            raise FileNotFoundError(f"JSON not found: {space_judgement_path}")

        features = maybe_load_json(features_path)
        audio_events = load_json(audio_events_path)
        space_similarity = maybe_load_json(space_similarity_path)
        space_judgement = load_json(space_judgement_path)

        payload = build_payload(
            features=features,
            audio_events=audio_events,
            space_judgement=space_judgement,
            space_similarity=space_similarity,
        )

        scene = generate_scene(payload)

        llm_debug = scene.pop("_debug_llm", {
            "enabled": LOCAL_LLM_ENABLED,
            "used": False,
            "status": "unknown",
            "reason": "debug info missing",
            "latency_sec": 0.0,
        })

        result = {
            **scene,
            "_meta": {
                "audio_id": audio_id,
                "output_file": str(output_path),
                "llm_enabled": llm_debug.get("enabled", False),
                "llm_used": llm_debug.get("used", False),
                "llm_status": llm_debug.get("status", "unknown"),
                "llm_reason": llm_debug.get("reason", ""),
                "llm_latency_sec": llm_debug.get("latency_sec", 0.0),
                "model": LOCAL_MODEL_NAME,
                "temperature": LOCAL_LLM_TEMPERATURE,
                "top_p": LOCAL_LLM_TOP_P,
            },
            "_input_summary": {
                "primary_space": payload.get("primary_space"),
                "secondary_space": payload.get("secondary_space"),
                "environment_family": payload.get("environment_family"),
                "audio_event_tags": payload.get("audio_event_tags"),
            },
        }

        save_json(output_path, result)
        print(f"[scene_interpreter] saved -> {output_path}")
        return result

    finally:
        clear_gpu_memory()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True, help="audio id")
    parser.add_argument("--pretty", action="store_true", help="pretty print json")
    args = parser.parse_args()

    try:
        out = run(args.audio_id)
        if args.pretty:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(out, ensure_ascii=False))
    finally:
        clear_gpu_memory()
