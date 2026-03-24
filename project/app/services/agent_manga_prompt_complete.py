#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_manga_prompt_v6_1.py

目的:
- v6 の「崩れにくい固定骨格」を維持
- 07_scene_interpretation.json の可変情報だけを LLM で反映
- Animagine XL 4.0 向けに、構図タグを主役にした prompt を生成
- 最終出力フォーマットは 08_manga_prompt.json 互換を維持

方針:
- 固定: 構図骨格 / must_keep / negative prompt / overlay
- LLM: title / narrative effects / seed compression / final compact
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/home/team-009/project")
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_LLM_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
ENV_LLM_ENABLED = os.getenv("MANGA_PROMPT_LLM", "1").strip().lower() in ("1", "true", "yes", "on")
ENV_LLM_MODEL = os.getenv("MANGA_PROMPT_LLM_MODEL", DEFAULT_LLM_MODEL)
ENV_MAX_NEW_TOKENS = int(os.getenv("MANGA_PROMPT_MAX_NEW_TOKENS", "260"))

OUTPUT_JSON_NAME = "08_manga_prompt.json"
DEBUG_DESIGN_NAME = "08_manga_design_debug.json"

NEGATIVE_PROMPT = (
    "text, watermark, logo, signature, low quality, worst quality, blurry, bad anatomy, "
    "bad hands, extra fingers, extra limbs, deformed, cropped, nsfw, background only, "
    "missing subject, tiny subject, out of frame, cut off body, multiple subjects, close-up, portrait, "
    "overcrowded city, giant buildings, futuristic megacity, distorted perspective, extreme perspective, "
    "flying structures, abstract architecture"
)

INPUT_FILES = [
    "07_scene_interpretation.json",
    "08_onomatopoeia.json",
]

_tokenizer = None
_model = None


# =========================================================
# basic utils
# =========================================================
def log(msg: str) -> None:
    print(f"[agent_manga_prompt_v6_1] {msg}", flush=True)


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


TAG_NORMALIZATION_MAP = {
    "open grass field": "grassy field",
    "urban edge ambience": "urban edge",
    "quiet atmosphere": "uneasy stillness",
    "small animal voice": "hidden presence",
    "animal presence": "small animal silhouette",
    "subtle subject": "small subject in frame",
    "black and white": "monochrome",
    "monochrome manga": "monochrome",
    "ink drawing": "inked lineart",
    "city at horizon edge": "urban edge",
    "city boundary": "urban edge",
}


def normalize_tag(tag: str) -> str:
    s = str(tag or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return TAG_NORMALIZATION_MAP.get(s, s)


def clamp_tags(tags: List[str], max_items: int = 32) -> List[str]:
    seen = set()
    out: List[str] = []
    for tag in tags:
        s = normalize_tag(tag)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def split_tags(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,\n;/]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def release_torch_memory(*objs: Any) -> None:
    global _model, _tokenizer

    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    # ここが重要
    _model = None
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

# =========================================================
# LLM core
# =========================================================
def load_llm() -> None:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    log(f"loading local llm: {ENV_LLM_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(ENV_LLM_MODEL, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        ENV_LLM_MODEL,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    _model.eval()
    _model.to("cuda" if torch.cuda.is_available() else "cpu")

    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token


def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 384) -> str:
    load_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(_tokenizer, "apply_chat_template"):
        prompt_text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    device = next(_model.parameters()).device
    inputs = _tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, ENV_MAX_NEW_TOKENS),
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            use_cache=False,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    release_torch_memory(inputs, output_ids, gen_ids)
    return text


def extract_json_block(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty response")

    for pattern in [
        r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON",
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))

    start = text.find("{")
    if start == -1:
        raise ValueError("json not found")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])

    raise ValueError("BEGIN_JSON/END_JSON block not found")


# =========================================================
# source facts
# =========================================================
def summarize_source_facts(source_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    scene = source_data.get("07_scene_interpretation.json") or {}
    ono = source_data.get("08_onomatopoeia.json") or {}

    facts = {
        "scene_title": str(scene.get("scene_title") or ""),
        "scene_summary": str(scene.get("scene_summary") or ""),
        "primary_space": str((scene.get("_input_summary") or {}).get("primary_space") or "unknown"),
        "mood_tags": [str(x).strip() for x in ensure_list(scene.get("mood_tags")) if str(x).strip()][:8],
        "visual_hints": [str(x).strip() for x in ensure_list(scene.get("visual_hints")) if str(x).strip()][:8],
        "audio_event_tags": [str(x).strip() for x in ensure_list(scene.get("audio_event_tags")) if str(x).strip()][:8],
        "timeline_summary": ensure_list(scene.get("timeline_summary"))[:8],
        "segment_audio_hints": ensure_list(scene.get("segment_audio_hints"))[:8],
        "narrative_hook": str(scene.get("narrative_hook") or ""),
        "manga_prompt_bridge": scene.get("manga_prompt_bridge") or {},
        "subject_hint": scene.get("subject_hint") or {},
        "onomatopoeia": {
            "primary": str(ono.get("primary_onomatopoeia") or ""),
            "intensity": str(ono.get("intensity") or ""),
            "style_hint": str(ono.get("style_hint") or ""),
            "placement_hint": str(ono.get("placement_hint") or ""),
        },
    }

    debug = []
    if facts["scene_title"]:
        debug.append(f"scene_title: {facts['scene_title']}")
    if facts["scene_summary"]:
        debug.append(f"scene_summary: {facts['scene_summary']}")
    if facts["primary_space"]:
        debug.append(f"primary_space: {facts['primary_space']}")
    if facts["mood_tags"]:
        debug.append("mood_tags: " + ", ".join(facts["mood_tags"][:5]))
    if facts["audio_event_tags"]:
        debug.append("audio_event_tags: " + ", ".join(facts["audio_event_tags"][:5]))
    if facts["narrative_hook"]:
        debug.append(f"narrative_hook: {facts['narrative_hook']}")
    return facts, debug


# =========================================================
# title refine (LLM)
# =========================================================
def build_default_title(facts: Dict[str, Any]) -> str:
    audio_tags = facts.get("audio_event_tags") or []
    title = str(facts.get("scene_title") or "").strip()
    if title and title not in {"不確かな風景", "曖昧な情景", "風景"}:
        return title
    if "Baby cry, infant cry" in audio_tags:
        return "遠くで泣く声"
    if "Cat" in audio_tags or "Animal" in audio_tags:
        return "見守る影"
    return title or "気配のある風景"


def refine_title_with_llm(facts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}
    base = build_default_title(facts)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        system = (
            "あなたは漫画タイトル作成AIです。"
            "出力は短いタイトル文字列のみ。説明・引用符・箇条書きは禁止です。"
            "抽象的すぎる『風景』『情景』は避けてください。"
        )
        user = (
            f"scene_summary: {facts.get('scene_summary')}\n"
            f"mood_tags: {', '.join(ensure_list(facts.get('mood_tags'))[:6])}\n"
            f"audio_event_tags: {', '.join(ensure_list(facts.get('audio_event_tags'))[:6])}\n"
            f"narrative_hook: {facts.get('narrative_hook')}\n"
            "短く、記憶に残る漫画タイトルを1つだけ返してください。"
        )
        raw = llm_generate(system, user, max_new_tokens=28)
        title = re.sub(r"[\r\n]+", " ", str(raw or "")).strip().strip('"').strip("'")
        if not title:
            raise ValueError("empty title")
        info.update({"used": True, "status": "success"})
        return title, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return base, info


# =========================================================
# structure extraction (fixed skeleton)
# =========================================================
def timeline_to_structure(facts: Dict[str, Any]) -> Dict[str, str]:
    rows = ensure_list(facts.get("timeline_summary"))[:8]
    foreground = "grassy hill"
    midground = "empty midground space"
    background = "urban edge"
    vertical_relation = "low horizon city"

    joined = " ".join(str(r.get("description") or "") for r in rows if isinstance(r, dict))
    if "草地" in joined or "開けた" in joined:
        foreground = "grassy hill"
    if "街" in joined or "都市" in joined:
        background = "urban edge"
        vertical_relation = "low horizon city"
    if "屋内" in joined:
        background = "thin skyline"

    return {
        "foreground": foreground,
        "midground": midground,
        "background": background,
        "vertical_relation": vertical_relation,
    }


# =========================================================
# narrative effects (LLM)
# =========================================================
def build_default_narrative_effects(facts: Dict[str, Any]) -> List[str]:
    hook = str(facts.get("narrative_hook") or "")
    tags: List[str] = []
    if "うごめいて" in hook:
        tags += ["subtle movement", "hidden presence", "uneasy stillness"]
    else:
        mood = ensure_list(facts.get("mood_tags"))
        if "不気味" in mood:
            tags.append("eerie atmosphere")
        if "不穏" in mood:
            tags.append("uneasy atmosphere")
        if "謎" in mood:
            tags.append("mysterious atmosphere")
    return clamp_tags(tags, 6)


def refine_narrative_effects_with_llm(facts: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}
    base = build_default_narrative_effects(facts)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        system = (
            "You convert a Japanese scene mood into 3 to 6 compact English image tags for manga atmosphere. "
            "Output only comma-separated English tags. No sentences."
        )
        user = (
            f"scene_summary: {facts.get('scene_summary')}\n"
            f"narrative_hook: {facts.get('narrative_hook')}\n"
            f"mood_tags: {', '.join(ensure_list(facts.get('mood_tags'))[:6])}\n"
            "Return compact image-effect tags for subtle unease, hidden presence, stillness, or related atmosphere."
        )
        raw = llm_generate(system, user, max_new_tokens=48)
        tags = clamp_tags(split_tags(raw), 8)
        if not tags:
            raise ValueError("empty effects")
        info.update({"used": True, "status": "success"})
        return tags, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return base, info


# =========================================================
# prompt_seed compression (LLM)
# =========================================================
def build_default_seed_tags(facts: Dict[str, Any]) -> List[str]:
    bridge = facts.get("manga_prompt_bridge") or {}
    prompt_seed = [normalize_tag(x) for x in ensure_list(bridge.get("prompt_seed_en")) if normalize_tag(x)]
    keep: List[str] = []
    for tag in prompt_seed:
        if any(k in tag for k in [
            "grass", "field", "urban", "city", "skyline", "animal", "quiet", "cry", "small", "subtle"
        ]):
            keep.append(tag)
    return clamp_tags(keep, 10)


def compress_seed_tags_with_llm(facts: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}
    base = build_default_seed_tags(facts)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        bridge = facts.get("manga_prompt_bridge") or {}
        system = (
            "You compress mixed English prompt seed phrases into compact Animagine-friendly English tags. "
            "Output only comma-separated English tags. Keep only image-effective tags."
        )
        user = (
            f"scene_summary: {facts.get('scene_summary')}\n"
            f"prompt_seed_en: {json.dumps(ensure_list(bridge.get('prompt_seed_en'))[:16], ensure_ascii=False)}\n"
            "Keep tags that help environment, small subject, subtle presence, urban edge, grass, skyline, hidden movement."
        )
        raw = llm_generate(system, user, max_new_tokens=80)
        tags = clamp_tags(split_tags(raw), 10)
        if not tags:
            raise ValueError("empty seed tags")
        info.update({"used": True, "status": "success"})
        return tags, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return base, info


# =========================================================
# composition JSON (LLM but bounded by fixed skeleton)
# =========================================================
COMPOSITION_REQUIRED_KEYS = [
    "camera_height",
    "subject_position",
    "subject_orientation",
    "background_position",
    "space_layout",
    "gaze_flow",
    "story_intent",
    "reason",
]


def validate_composition_json(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("composition json must be dict")
    missing = [k for k in COMPOSITION_REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"composition json missing keys: {missing}")


def build_default_composition(facts: Dict[str, Any]) -> Dict[str, Any]:
    subject_hint = facts.get("subject_hint") or {}
    subject_type = str(subject_hint.get("subject_type") or "none").strip()

    primary = str(facts.get("primary_space") or "unknown").lower()
    summary = str(facts.get("scene_summary") or "")
    visual_hints = [str(x) for x in ensure_list(facts.get("visual_hints"))]
    audio_tags = [str(x).lower() for x in ensure_list(facts.get("audio_event_tags"))]

    inside_score = 0
    outside_score = 0

    near_human_machine = {"speech", "vehicle", "door", "sliding door", "train", "music", "shuffle"}
    ambient_open = {"wind", "wind noise (microphone)", "rustling leaves", "animal", "birds", "bird", "water", "river"}

    if any(tag in near_human_machine for tag in audio_tags):
        inside_score += 2
    if any(tag in ambient_open for tag in audio_tags):
        outside_score += 2

    joined_visual = " ".join(visual_hints) + " " + summary
    if any(word in joined_visual for word in ["交差点", "通り", "車", "歩行者", "ネオン", "看板", "駅", "室内", "店内"]):
        inside_score += 1
    if any(word in joined_visual for word in ["遠く", "見渡", "地平", "丘", "山", "川辺", "森", "空"]):
        outside_score += 1

    if primary in {"urban", "urban_street", "indoor", "indoor_room"}:
        inside_score += 1
    if primary in {"forest", "dense_forest", "river_side", "mountain_slope", "open_field", "open_nature"}:
        outside_score += 1

    orientation = "side_silhouette" if subject_type == "animal" else ("natural" if subject_type == "human" else "natural")

    if inside_score >= outside_score:
        return {
            "camera_height": "eye",
            "subject_position": "mid",
            "subject_orientation": orientation,
            "background_position": "layered_behind_subject",
            "space_layout": "dense_layered_space",
            "gaze_flow": "into_scene",
            "story_intent": "immersion",
            "reason": f"fallback composition derived from audio proximity and primary_space={primary}",
        }

    return {
        "camera_height": "high",
        "subject_position": "foreground_right",
        "subject_orientation": "side_silhouette" if subject_type == "animal" else "back_view",
        "background_position": "far_back",
        "space_layout": "medium_negative_space",
        "gaze_flow": "toward_depth",
        "story_intent": "observation",
        "reason": f"fallback composition derived from ambient spread and primary_space={primary}",
    }


def refine_composition_with_llm(facts: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": "", "raw_preview": ""}
    base = build_default_composition(facts)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        system = (
            "You output only composition JSON for a manga image. "
            "Infer viewpoint from audio-derived scene facts, not from a fixed template. "
            "Choose whether the viewer feels inside the scene or observing from outside. "
            "Use BEGIN_JSON and END_JSON. Output one JSON object only."
        )
        user = f"""facts:
{json.dumps({
    "scene_summary": facts.get("scene_summary"),
    "primary_space": facts.get("primary_space"),
    "mood_tags": ensure_list(facts.get("mood_tags"))[:6],
    "visual_hints": ensure_list(facts.get("visual_hints"))[:6],
    "audio_event_tags": ensure_list(facts.get("audio_event_tags"))[:8],
    "subject_hint": facts.get("subject_hint"),
    "timeline_summary": ensure_list(facts.get("timeline_summary"))[:6],
}, ensure_ascii=False, indent=2)}

Rules:
- Return valid JSON only between BEGIN_JSON and END_JSON.
- Decide viewpoint generically from the facts.
- If the sound suggests the listener is inside a dense scene, prefer immersion.
- If the sound suggests open ambient spread, prefer observation.
- Do not force hillside overlook unless the facts clearly imply it.
- Keep values compact and image-oriented.

Allowed values guidance:
- camera_height: eye / mid / high
- subject_position: foreground_left / foreground_right / mid / center
- subject_orientation: natural / side_silhouette / side_view / back_view
- background_position: layered_behind_subject / mid_back / far_back / far_below
- space_layout: dense_layered_space / balanced_depth / medium_negative_space / large_negative_space
- gaze_flow: into_scene / toward_subject / toward_depth / toward_background
- story_intent: immersion / observation / uneasy_observation / quiet_observation

Return this schema:
BEGIN_JSON
{{
  "camera_height": "eye",
  "subject_position": "mid",
  "subject_orientation": "natural",
  "background_position": "layered_behind_subject",
  "space_layout": "balanced_depth",
  "gaze_flow": "into_scene",
  "story_intent": "immersion",
  "reason": "short japanese sentence"
}}
END_JSON"""
        raw = llm_generate(system, user, max_new_tokens=180)
        info["raw_preview"] = raw[:1000]
        obj = extract_json_block(raw)
        validate_composition_json(obj)
        merged = dict(base)
        for k in COMPOSITION_REQUIRED_KEYS:
            if obj.get(k) not in (None, ""):
                merged[k] = obj[k]
        info.update({"used": True, "status": "success"})
        return merged, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return base, info


# =========================================================
# composition tags (fixed mapping + LLM support)
# =========================================================
def composition_to_tags(comp: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    camera_height = str(comp.get("camera_height") or "")
    subject_position = str(comp.get("subject_position") or "")
    background_position = str(comp.get("background_position") or "")
    space_layout = str(comp.get("space_layout") or "")
    subject_orientation = str(comp.get("subject_orientation") or "")
    gaze_flow = str(comp.get("gaze_flow") or "")

    if camera_height == "high":
        tags += ["high angle"]
    elif camera_height == "eye":
        tags += ["eye level"]
    elif camera_height == "mid":
        tags += ["mid level view"]

    if subject_position == "foreground_right":
        tags += ["foreground", "right side"]
    elif subject_position == "foreground_left":
        tags += ["foreground", "left side"]
    elif subject_position == "mid":
        tags += ["mid frame"]
    elif subject_position == "center":
        tags += ["center composition"]

    if background_position == "far_below":
        tags += ["city below", "low horizon city"]
    elif background_position == "horizon_far_back":
        tags += ["far horizon", "thin skyline"]
    elif background_position == "far_back":
        tags += ["deep background"]
    elif background_position == "mid_back":
        tags += ["layered background"]
    elif background_position == "layered_behind_subject":
        tags += ["layered background", "depth separation"]

    if space_layout == "medium_negative_space":
        tags += ["negative space", "clear separation"]
    elif space_layout == "large_negative_space":
        tags += ["negative space", "empty space", "clear separation"]
    elif space_layout == "dense_layered_space":
        tags += ["dense layered scene", "immersive perspective"]
    elif space_layout == "balanced_depth":
        tags += ["balanced depth", "depth separation"]

    if subject_orientation == "side_silhouette":
        tags += ["side silhouette"]
    elif subject_orientation == "back_view":
        tags += ["from behind"]
    elif subject_orientation == "side_view":
        tags += ["side view"]

    if gaze_flow == "toward_background":
        tags += ["looking into distance"]
    elif gaze_flow == "toward_depth":
        tags += ["looking into depth"]
    elif gaze_flow == "into_scene":
        tags += ["within environment", "into scene"]
    elif gaze_flow == "toward_subject":
        tags += ["subject emphasis"]

    return clamp_tags(tags, 16)


def refine_composition_tags_with_llm(comp: Dict[str, Any], facts: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}
    base = composition_to_tags(comp)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        system = (
            "You convert composition to compact Animagine-friendly tags. "
            "Output only comma-separated English tags. Preserve vertical relation and negative space."
        )
        user = (
            f"composition: {json.dumps(comp, ensure_ascii=False)}\n"
            f"scene_summary: {facts.get('scene_summary')}\n"
            "Return only compact tags."
        )
        raw = llm_generate(system, user, max_new_tokens=60)
        tags = clamp_tags(split_tags(raw), 16)
        if not tags:
            raise ValueError("empty composition tags")
        # merge with base to keep must-have tags
        merged = []
        seen = set()
        for t in tags + base:
            n = normalize_tag(t)
            if n and n not in seen:
                merged.append(n)
                seen.add(n)
        info.update({"used": True, "status": "success"})
        return clamp_tags(merged, 16), info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return base, info


# =========================================================
# prompt assembly
# =========================================================
def build_fixed_skeleton_tags(facts: Dict[str, Any], composition: Dict[str, Any]) -> List[str]:
    structure = timeline_to_structure(facts)
    subject_hint = facts.get("subject_hint") or {}
    subject_type = str(subject_hint.get("subject_type") or "none")

    tags = [
        "masterpiece",
        "best quality",
        "anime style",
        "manga style",
        "2d illustration",
        "monochrome",
        "manga screentone",
        "inked lineart",
        "high contrast",
        "wide shot",
    ]

    tags += composition_to_tags(composition)

    if subject_type == "animal":
        tags += ["small animal silhouette", "small subject in frame", "long shadow"]
    elif subject_type == "human":
        tags += ["1 person", "small subject in frame", "long shadow"]

    tags += [
        structure["foreground"],
        structure["midground"],
        structure["background"],
        structure["vertical_relation"],
    ]

    return clamp_tags(tags, 24)



def clean_composition_tags(tags: List[str]) -> List[str]:
    fixed: List[str] = []
    for t in ensure_list(tags):
        n = normalize_tag(t)
        if not n:
            continue
        if n == "high":
            n = "high angle"
        if n == "foreground left":
            fixed.extend(["foreground", "left side"])
            continue
        if n == "foreground right":
            fixed.extend(["foreground", "right side"])
            continue
        if n == "far below":
            fixed.extend(["city below", "low horizon city"])
            continue
        if n == "medium negative space":
            fixed.extend(["negative space", "clear separation"])
            continue
        if n == "dense layered space":
            fixed.extend(["dense layered scene", "immersive perspective"])
            continue
        if n == "balanced depth":
            fixed.extend(["balanced depth", "depth separation"])
            continue
        if any(x in n for x in ["sound youthful", "sound animal", "background nature", "background cityscape"]):
            continue
        fixed.append(n)
    return clamp_tags(fixed, 16)


def compress_effect_tags(tags: List[str]) -> List[str]:
    priority = [
        "hidden presence",
        "subtle movement",
        "uneasy stillness",
        "eerie",
        "unsettling",
        "hushed",
        "lurking",
    ]
    normalized = [normalize_tag(x) for x in ensure_list(tags) if normalize_tag(x)]
    out: List[str] = []
    for p in priority:
        if p in normalized and p not in out:
            out.append(p)
    return clamp_tags(out, 4)


def resolve_position_conflict(tags: List[str], composition: Dict[str, Any]) -> List[str]:
    pos = str(composition.get("subject_position") or "").strip()
    normalized = [normalize_tag(x) for x in ensure_list(tags) if normalize_tag(x)]
    if pos == "foreground_left":
        normalized = [t for t in normalized if t != "right side"]
    elif pos == "foreground_right":
        normalized = [t for t in normalized if t != "left side"]
    return clamp_tags(normalized, 32)


def dedupe_city_background(tags: List[str]) -> List[str]:
    normalized = [normalize_tag(x) for x in ensure_list(tags) if normalize_tag(x)]
    out: List[str] = []
    seen = set()
    city_family = {"city below", "low horizon city", "far horizon", "thin skyline", "urban edge"}
    city_used = 0
    for t in normalized:
        if t in city_family:
            city_used += 1
            if city_used > 3:
                continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    return clamp_tags(out, 32)


def merge_core_tags(compact_tags: List[str], must_keep: List[str], extras: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for tag in compact_tags + must_keep + extras:
        n = normalize_tag(tag)
        if n and n not in seen:
            merged.append(n)
            seen.add(n)
    merged = dedupe_city_background(merged)
    return clamp_tags(merged, 28)


def compact_with_llm(skeleton_tags: List[str], effect_tags: List[str], seed_tags: List[str], composition_tags: List[str], composition: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}

    composition_tags = clean_composition_tags(composition_tags)
    effect_tags = compress_effect_tags(effect_tags)
    seed_tags = clamp_tags([normalize_tag(x) for x in ensure_list(seed_tags)], 4)

    must_keep = [
        "wide shot",
        "monochrome", "manga screentone", "inked lineart",
        "small subject in frame"
    ]
    for t in composition_tags:
        if t in {
            "high angle", "eye level", "mid level view",
            "foreground", "left side", "right side", "mid frame", "center composition",
            "negative space", "clear separation",
            "city below", "low horizon city", "layered background", "depth separation",
            "dense layered scene", "immersive perspective", "within environment", "into scene",
            "looking into distance", "looking into depth"
        }:
            must_keep.append(t)
    pos = str(composition.get("subject_position") or "").strip()
    if pos == "foreground_left":
        must_keep.append("left side")
    elif pos == "foreground_right":
        must_keep.append("right side")

    fallback = merge_core_tags(skeleton_tags, must_keep, effect_tags + seed_tags + composition_tags)
    fallback = resolve_position_conflict(fallback, composition)

    if not ENV_LLM_ENABLED:
        return fallback, info

    try:
        system = (
            "You are an Animagine XL 4.0 prompt optimizer. "
            "Output only comma-separated English tags. "
            "Keep composition-critical tags. Reduce weak atmosphere tags. "
            "Never keep both left side and right side together. "
            "If high angle is present, preserve vertical relation like city below or low horizon city."
        )
        user = (
            f"skeleton_tags: {', '.join(skeleton_tags)}\n"
            f"effect_tags: {', '.join(effect_tags)}\n"
            f"seed_tags: {', '.join(seed_tags)}\n"
            f"composition_tags: {', '.join(composition_tags)}\n"
            f"subject_position: {composition.get('subject_position')}\n"
            "Return the best compact final tags for Animagine. "
            "Preserve composition-critical tags derived from composition_tags and must_keep. "
            "Keep only one side tag."
        )
        raw = llm_generate(system, user, max_new_tokens=90)
        compact_tags = clamp_tags(split_tags(raw), 24)
        if not compact_tags:
            raise ValueError("empty compact tags")
        final_tags = merge_core_tags(compact_tags, must_keep, effect_tags + seed_tags + composition_tags)
        final_tags = resolve_position_conflict(final_tags, composition)
        info.update({"used": True, "status": "success"})
        return final_tags, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return fallback, info

    try:
        system = (
            "You are an Animagine XL 4.0 prompt optimizer. "
            "Output only comma-separated English tags. "
            "Keep composition-critical tags. Reduce weak atmosphere tags."
        )
        user = (
            f"skeleton_tags: {', '.join(skeleton_tags)}\n"
            f"effect_tags: {', '.join(effect_tags)}\n"
            f"seed_tags: {', '.join(seed_tags)}\n"
            f"composition_tags: {', '.join(composition_tags)}\n"
            "Return the best compact final tags for Animagine. "
            "Do not remove high angle, foreground, right side, city below, low horizon city, negative space, clear separation."
        )
        raw = llm_generate(system, user, max_new_tokens=90)
        compact_tags = clamp_tags(split_tags(raw), 24)
        if not compact_tags:
            raise ValueError("empty compact tags")
        info.update({"used": True, "status": "success"})
        return merge_core_tags(compact_tags, must_keep, effect_tags + seed_tags + composition_tags), info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return merge_core_tags(skeleton_tags, must_keep, effect_tags + seed_tags + composition_tags), info


# =========================================================
# output
# =========================================================
def build_output(audio_id: str, facts: Dict[str, Any], final_tags: List[str], composition: Dict[str, Any], title: str) -> Dict[str, Any]:
    onomatopoeia = facts.get("onomatopoeia") or {}
    panel_plan = [{
        "panel_index": 1,
        "shot": "wide",
        "angle": str(composition.get("subject_orientation") or "side_silhouette"),
        "composition": "wide_shot",
        "subject_present": True,
        "subject_scale": "small",
        "focus_priority": "environment",
    }]
    ono_layout = []
    if onomatopoeia.get("primary"):
        ono_layout = [{
            "text": str(onomatopoeia.get("primary") or ""),
            "mode": "overlay",
            "anchor": "diagonal",
            "intensity": str(onomatopoeia.get("intensity") or "medium"),
        }]

    return {
        "audio_id": audio_id,
        "title": title,
        "manga_title": title,
        "scene_match": str(facts.get("primary_space") or ""),
        "confidence": 0.8,
        "panel_count": 1,
        "panel_plan": panel_plan,
        "positive_prompt": ", ".join(final_tags),
        "negative_prompt": NEGATIVE_PROMPT,
        "onomatopoeia_layout": ono_layout,
        "direction_notes": [
            "v6.1 fixed skeleton + llm variable refinement",
            "Animagine-optimized compact tags",
            "onomatopoeia should be overlaid after image generation",
        ],
        "llm": {
            "enabled": ENV_LLM_ENABLED,
            "used": ENV_LLM_ENABLED,
            "status": "success" if ENV_LLM_ENABLED else "disabled",
            "model": ENV_LLM_MODEL,
            "reason": "title/effects/seed/composition/compact refined on top of fixed skeleton",
        },
        "source_summary": {
            "scene_title": facts.get("scene_title") or "",
            "primary_environment": facts.get("primary_space") or "",
            "mood_tags": facts.get("mood_tags") or [],
            "subject_hint": facts.get("subject_hint") or {},
        },
    }


# =========================================================
# pipeline
# =========================================================
def run_pipeline(audio_id: str) -> Dict[str, Any]:
    result_dir = RESULTS_DIR / audio_id
    source = {name: safe_read_json(result_dir / name) for name in INPUT_FILES}
    if not source.get("07_scene_interpretation.json"):
        raise FileNotFoundError(f"missing required input: {result_dir / '07_scene_interpretation.json'}")

    facts, debug_facts = summarize_source_facts(source)
    title, title_llm = refine_title_with_llm(facts)
    facts["refined_title"] = title

    effect_tags, effects_llm = refine_narrative_effects_with_llm(facts)
    seed_tags, seed_llm = compress_seed_tags_with_llm(facts)
    composition, composition_llm = refine_composition_with_llm(facts)
    composition_tags, composition_tags_llm = refine_composition_tags_with_llm(composition, facts)

    skeleton_tags = build_fixed_skeleton_tags(facts, composition)
    final_tags, compact_llm = compact_with_llm(skeleton_tags, effect_tags, seed_tags, composition_tags, composition)

    output = build_output(audio_id, facts, final_tags, composition, title)
    output["debug_facts"] = debug_facts
    output["debug_design"] = {
        "composition": composition,
        "composition_tags": composition_tags,
        "effect_tags": effect_tags,
        "seed_tags": seed_tags,
        "skeleton_tags": skeleton_tags,
        "final_tags": final_tags,
    }

    debug_payload = {
        "audio_id": audio_id,
        "facts": facts,
        "composition": composition,
        "composition_tags": composition_tags,
        "effect_tags": effect_tags,
        "seed_tags": seed_tags,
        "skeleton_tags": skeleton_tags,
        "final_tags": final_tags,
        "title_llm": title_llm,
        "effects_llm": effects_llm,
        "seed_llm": seed_llm,
        "composition_llm": composition_llm,
        "composition_tags_llm": composition_tags_llm,
        "compact_llm": compact_llm,
    }

    save_json(result_dir / "08_manga_prompt.json", output)
    save_json(result_dir / "08_manga_design_debug.json", debug_payload)

    log(f"Saved -> {result_dir / '08_manga_prompt.json'}")
    log(f"Saved -> {result_dir / '08_manga_design_debug.json'}")
    return output

def run(audio_id: str, pretty: bool = True):
    try:
        return run_pipeline(audio_id)
    except Exception as e:
        log(f"run failed: {e}")
        return {
            "audio_id": audio_id,
            "error": str(e),
            "status": "failed"
        }
    finally:
        release_torch_memory(_model, _tokenizer)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        result = run_pipeline(args.audio_id)
        if args.pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        traceback.print_exc()
    finally:
        release_torch_memory(_model, _tokenizer)


if __name__ == "__main__":
    main()
