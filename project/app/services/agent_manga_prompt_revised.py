#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_manga_prompt_revised.py

目的:
- 既存パイプライン / 入出力フォーマットを壊さずに 08_manga_prompt.json を生成
- 07_scene_interpretation.json と 08_onomatopoeia.json を主入力として使用
- 必要に応じて 06_space_judgement.json を補助参照するが、未存在でも動作する
- LLM による汎用的な描画方針生成を強化しつつ、後段互換を維持する

主な改善点:
- 固定の「丘 + 都市見下ろし」骨格を廃止し、入力データから構図方針を決定
- urban / indoor / river / forest / field / mountain などを汎用的に扱う
- LLM 出力を JSON で拘束しつつ、失敗時は deterministic fallback へ退避
- 出力キーは既存 08_manga_prompt.json 互換を維持
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

# 既存互換のため名前は維持。ただし内容は「普通の都市」を潰しすぎないように調整。
NEGATIVE_PROMPT = (
    "text, watermark, logo, signature, low quality, worst quality, blurry, bad anatomy, "
    "bad hands, extra fingers, extra limbs, deformed, cropped, nsfw, background only, "
    "missing subject, tiny subject, out of frame, cut off body, multiple subjects, close-up, portrait, "
    "futuristic megacity, distorted perspective, extreme perspective, flying structures, abstract architecture"
)

# 必須入力は従来通り 07 / 08。06 は任意。
INPUT_FILES = [
    "07_scene_interpretation.json",
    "08_onomatopoeia.json",
    "06_space_judgement.json",
]

_tokenizer = None
_model = None


# =========================================================
# basic utils
# =========================================================
def log(msg: str) -> None:
    print(f"[agent_manga_prompt_revised] {msg}", flush=True)



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
    "black and white": "monochrome",
    "monochrome manga": "monochrome",
    "ink drawing": "inked lineart",
    "1 human": "1 person",
    "human silhouette": "person silhouette",
    "small character": "small subject in frame",
    "subtle subject": "small subject in frame",
    "cityscape as main focus": "cityscape as main subject",
    "urban main subject": "cityscape as main subject",
    "forest main subject": "forest as main subject",
    "riverside main subject": "riverside as main subject",
    "mountain main subject": "mountain landscape as main subject",
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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    _model.eval()
    _model.to("cuda" if torch.cuda.is_available() else "cpu")

    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token



def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 180) -> str:
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

    def _extract_balanced_json(s: str) -> Dict[str, Any]:
        start = s.find("{")
        if start == -1:
            raise ValueError("json not found")

        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(s[start:], start=start):
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
                    return json.loads(s[start:i + 1])

        raise ValueError("balanced json not found")

    # 1) BEGIN_JSON ... END_JSON を優先
    m = re.search(
        r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return json.loads(m.group(1))

    # 2) BEGIN_JSON があれば、END_JSON がなくても以降を救済
    m = re.search(r"BEGIN_JSON\s*(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return _extract_balanced_json(candidate)
        except Exception:
            pass

    # 3) コードフェンス内 JSON
    for pattern in [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))

    # 4) 全文から最初の balanced JSON を探す
    try:
        return _extract_balanced_json(text)
    except Exception:
        pass

    raise ValueError("json not found")

# =========================================================
# source facts
# =========================================================
def infer_environment_family(primary_space: str, scene: Dict[str, Any], judgement: Dict[str, Any]) -> str:
    p = str(primary_space or "").lower()
    if "urban" in p:
        return "urban"
    if "indoor" in p or "room" in p:
        return "indoor"
    if "river" in p or "water" in p:
        return "waterside"
    if "forest" in p or "wood" in p:
        return "forest"
    if "mountain" in p or "slope" in p:
        return "mountain"
    if "field" in p or "grass" in p or "open" in p:
        return "open_nature"

    env_tags = " ".join(str(x) for x in ensure_list(scene.get("environment_tags")))
    if "都市" in env_tags:
        return "urban"
    if "屋内" in env_tags:
        return "indoor"
    if any(k in env_tags for k in ["川", "水辺", "海"]):
        return "waterside"
    return str(judgement.get("environment_family") or "unknown") or "unknown"



def summarize_source_facts(source_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    scene = source_data.get("07_scene_interpretation.json") or {}
    ono = source_data.get("08_onomatopoeia.json") or {}
    judgement = source_data.get("06_space_judgement.json") or {}
    input_summary = scene.get("_input_summary") or {}

    primary_space = str(input_summary.get("primary_space") or judgement.get("attributes", {}).get("primary_space") or "unknown")
    environment_family = infer_environment_family(primary_space, scene, judgement)

    facts = {
        "scene_title": str(scene.get("scene_title") or ""),
        "scene_summary": str(scene.get("scene_summary") or ""),
        "primary_space": primary_space,
        "secondary_space": str(judgement.get("attributes", {}).get("secondary_space") or ""),
        "final_space_label": str(judgement.get("final_space_label") or ""),
        "environment_family": environment_family,
        "mood_tags": [str(x).strip() for x in ensure_list(scene.get("mood_tags")) if str(x).strip()][:8],
        "environment_tags": [str(x).strip() for x in ensure_list(scene.get("environment_tags")) if str(x).strip()][:8],
        "visual_hints": [str(x).strip() for x in ensure_list(scene.get("visual_hints")) if str(x).strip()][:10],
        "audio_event_tags": [str(x).strip() for x in ensure_list(scene.get("audio_event_tags")) if str(x).strip()][:10],
        "timeline_summary": ensure_list(scene.get("timeline_summary"))[:10],
        "segment_audio_hints": ensure_list(scene.get("segment_audio_hints"))[:10],
        "narrative_hook": str(scene.get("narrative_hook") or ""),
        "manga_prompt_bridge": scene.get("manga_prompt_bridge") or {},
        "subject_hint": scene.get("subject_hint") or {},
        "judgement_reason": [str(x).strip() for x in ensure_list(judgement.get("reason")) if str(x).strip()][:8],
        "judgement_confidence": float(judgement.get("confidence") or 0.0),
        "onomatopoeia": {
            "primary": str(ono.get("primary_onomatopoeia") or ""),
            "intensity": str(ono.get("intensity") or ""),
            "style_hint": str(ono.get("style_hint") or ""),
            "placement_hint": str(ono.get("placement_hint") or ""),
        },
    }

    debug = []
    for key in [
        "scene_title", "scene_summary", "primary_space", "secondary_space",
        "environment_family", "final_space_label", "narrative_hook"
    ]:
        if facts.get(key):
            debug.append(f"{key}: {facts[key]}")
    if facts["mood_tags"]:
        debug.append("mood_tags: " + ", ".join(facts["mood_tags"][:5]))
    if facts["environment_tags"]:
        debug.append("environment_tags: " + ", ".join(facts["environment_tags"][:5]))
    if facts["audio_event_tags"]:
        debug.append("audio_event_tags: " + ", ".join(facts["audio_event_tags"][:5]))
    return facts, debug


# =========================================================
# title refine
# =========================================================
def build_default_title(facts: Dict[str, Any]) -> str:
    title = str(facts.get("scene_title") or "").strip()
    if title and title not in {"不確かな風景", "曖昧な情景", "風景", "情景"}:
        return title

    family = str(facts.get("environment_family") or "unknown")
    if family == "urban":
        return "街の気配"
    if family == "indoor":
        return "室内の余韻"
    if family == "waterside":
        return "水辺の気配"
    if family == "forest":
        return "森の気配"
    if family == "mountain":
        return "斜面の向こう"
    return title or "気配のある風景"



def refine_title_with_llm(facts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}
    base = build_default_title(facts)
    if not ENV_LLM_ENABLED:
        return base, info
    try:
        system = (
            "あなたは漫画タイトル作成AIです。出力は短いタイトル文字列のみ。"
            "説明・引用符・括弧・番号は禁止です。"
            "場面の主空間を外さないでください。"
        )
        user = (
            f"scene_summary: {facts.get('scene_summary')}\n"
            f"primary_space: {facts.get('primary_space')}\n"
            f"environment_family: {facts.get('environment_family')}\n"
            f"mood_tags: {', '.join(ensure_list(facts.get('mood_tags'))[:6])}\n"
            f"narrative_hook: {facts.get('narrative_hook')}\n"
            "短く、記憶に残る日本語タイトルを1つだけ返してください。"
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
# generic deterministic fallback design
# =========================================================
def build_environment_tokens(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    family = str(facts.get("environment_family") or "unknown")
    primary = str(facts.get("primary_space") or "")

    common = ["wide shot", "cinematic composition", "depth", "atmospheric perspective"]

    if family == "urban":
        return {
            "subject": ["cityscape as main subject", "urban environment dominant", "detailed skyline", "visible streets"],
            "foreground": ["foreground depth"],
            "midground": ["layered buildings"],
            "background": ["city lights" if "night" in facts.get("scene_summary", "").lower() else "expansive city skyline"],
            "common": common,
        }
    if family == "indoor":
        return {
            "subject": ["interior space as main subject", "room atmosphere", "architectural interior"],
            "foreground": ["foreground object silhouette"],
            "midground": ["open floor space"],
            "background": ["room depth"],
            "common": common,
        }
    if family == "waterside":
        return {
            "subject": ["riverside as main subject", "water surface", "waterside atmosphere"],
            "foreground": ["foreground grass"],
            "midground": ["shoreline depth"],
            "background": ["distant waterline"],
            "common": common,
        }
    if family == "forest":
        return {
            "subject": ["forest as main subject", "layered trees", "woodland atmosphere"],
            "foreground": ["foreground foliage"],
            "midground": ["tree depth"],
            "background": ["forest distance haze"],
            "common": common,
        }
    if family == "mountain":
        return {
            "subject": ["mountain landscape as main subject", "sloped terrain", "ridge lines"],
            "foreground": ["foreground slope"],
            "midground": ["terrain layers"],
            "background": ["distant mountains"],
            "common": common,
        }
    return {
        "subject": ["landscape as main subject", "environment dominant"],
        "foreground": ["foreground depth"],
        "midground": ["midground separation"],
        "background": ["distant background"],
        "common": common,
    }



def choose_subject_orientation(facts: Dict[str, Any]) -> str:
    subject_hint = facts.get("subject_hint") or {}
    framing = str(subject_hint.get("framing_hint") or "").strip().lower()
    subject_type = str(subject_hint.get("subject_type") or "").strip().lower()
    if framing in {"side_view", "side_silhouette"}:
        return "side_view"
    if framing in {"back_view", "from_behind"}:
        return "back_view"
    if subject_type == "human":
        return "back_view"
    if subject_type in {"animal", "bird"}:
        return "side_view"
    return "back_view"



def build_default_composition(facts: Dict[str, Any]) -> Dict[str, Any]:
    family = str(facts.get("environment_family") or "unknown")
    orientation = choose_subject_orientation(facts)
    subject_hint = facts.get("subject_hint") or {}
    has_subject = str(subject_hint.get("subject_type") or "").strip() not in {"", "none", "unknown"}

    camera_height = "eye_level"
    subject_position = "midground_center"
    background_position = "far_back"
    space_layout = "balanced_layers"
    gaze_flow = "toward_depth"
    story_intent = "observation"

    if family in {"urban", "mountain", "open_nature", "waterside", "forest"}:
        camera_height = "high" if family in {"urban", "mountain"} else "eye_level"
        background_position = "far_back"
        space_layout = "layered_depth"

    if has_subject:
        subject_position = "foreground_left" if family in {"urban", "mountain"} else "foreground_right"

    return {
        "camera_height": camera_height,
        "subject_position": subject_position,
        "subject_orientation": orientation,
        "background_position": background_position,
        "space_layout": space_layout,
        "gaze_flow": gaze_flow,
        "story_intent": story_intent,
        "reason": f"fallback composition derived from environment_family={family}",
    }


# =========================================================
# bounded llm design JSON
# =========================================================
DESIGN_REQUIRED_KEYS = [
    "title",
    "composition",
    "effect_tags",
    "seed_tags",
    "compact_tags",
]

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



def validate_design_json(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("design json must be dict")
    missing = [k for k in DESIGN_REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"design json missing keys: {missing}")
    comp = obj.get("composition")
    if not isinstance(comp, dict):
        raise ValueError("composition must be dict")
    comp_missing = [k for k in COMPOSITION_REQUIRED_KEYS if k not in comp]
    if comp_missing:
        raise ValueError(f"composition missing keys: {comp_missing}")



def refine_design_with_llm(facts: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": "", "raw_preview": ""}

    fallback_title = build_default_title(facts)
    fallback_composition = build_default_composition(facts)
    env_tokens = build_environment_tokens(facts)
    fallback_effects = clamp_tags([
        "quiet atmosphere" if facts.get("environment_family") != "urban" else "urban ambience",
        "subtle mood",
    ], 4)
    fallback_seed = clamp_tags(
        env_tokens["subject"] + env_tokens["foreground"] + env_tokens["midground"] + env_tokens["background"],
        10,
    )
    fallback_compact = clamp_tags(
        env_tokens["common"] + env_tokens["subject"] + fallback_seed[:4],
        12,
    )

    fallback = {
        "title": fallback_title,
        "composition": fallback_composition,
        "effect_tags": fallback_effects,
        "seed_tags": fallback_seed,
        "compact_tags": fallback_compact,
    }

    if not ENV_LLM_ENABLED:
        return fallback, info

    try:
        system = (
            "You are a manga prompt design planner for image generation. "
            "Read the input facts and output only JSON. "
            "Keep the environment faithful to the input. "
            "Do not force hillside, grass, city-below, or rural scenery unless the facts support them. "
            "The output must be generic and reusable across urban, indoor, waterside, forest, field, and mountain scenes."
        )
        user = f"""facts:
{json.dumps({
    "scene_title": facts.get("scene_title"),
    "scene_summary": facts.get("scene_summary"),
    "primary_space": facts.get("primary_space"),
    "secondary_space": facts.get("secondary_space"),
    "final_space_label": facts.get("final_space_label"),
    "environment_family": facts.get("environment_family"),
    "mood_tags": facts.get("mood_tags"),
    "environment_tags": facts.get("environment_tags"),
    "visual_hints": facts.get("visual_hints"),
    "audio_event_tags": facts.get("audio_event_tags"),
    "narrative_hook": facts.get("narrative_hook"),
    "subject_hint": facts.get("subject_hint"),
    "onomatopoeia": facts.get("onomatopoeia"),
    "judgement_reason": facts.get("judgement_reason"),
}, ensure_ascii=False, indent=2)}

constraints:
- output must be BEGIN_JSON ... END_JSON
- title: short Japanese manga title string
- composition: JSON with keys {COMPOSITION_REQUIRED_KEYS}
- effect_tags: 2 to 6 compact English tags for atmosphere
- seed_tags: 4 to 10 compact English tags reflecting environment and depth
- compact_tags: 6 to 14 compact English tags suitable for final prompt compression
- avoid unsupported claims
- if urban is primary, do not weaken city into only distant tiny background
- if indoor is primary, do not turn it into outdoor
- if waterside / forest / mountain / field is primary, do not inject city unless supported

format:
BEGIN_JSON
{json.dumps(fallback, ensure_ascii=False, indent=2)}
END_JSON"""
        raw = llm_generate(system, user, max_new_tokens=220)
        info["raw_preview"] = raw[:1500]
        obj = extract_json_block(raw)
        validate_design_json(obj)

        design = {
            "title": str(obj.get("title") or fallback_title).strip() or fallback_title,
            "composition": dict(fallback_composition),
            "effect_tags": clamp_tags(ensure_list(obj.get("effect_tags")), 6) or fallback_effects,
            "seed_tags": clamp_tags(ensure_list(obj.get("seed_tags")), 10) or fallback_seed,
            "compact_tags": clamp_tags(ensure_list(obj.get("compact_tags")), 14) or fallback_compact,
        }

        llm_comp = obj.get("composition") or {}
        for k in COMPOSITION_REQUIRED_KEYS:
            v = llm_comp.get(k)
            if v not in (None, ""):
                design["composition"][k] = v

        info.update({"used": True, "status": "success"})
        return design, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return fallback, info


# =========================================================
# tag synthesis
# =========================================================
def composition_to_tags(comp: Dict[str, Any], facts: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    cam = str(comp.get("camera_height") or "")
    pos = str(comp.get("subject_position") or "")
    orient = str(comp.get("subject_orientation") or "")
    bg = str(comp.get("background_position") or "")
    layout = str(comp.get("space_layout") or "")
    gaze = str(comp.get("gaze_flow") or "")
    family = str(facts.get("environment_family") or "unknown")

    if cam == "high":
        tags.append("high angle")
    elif cam == "low":
        tags.append("low angle")
    else:
        tags.append("eye level")

    if pos == "foreground_left":
        tags += ["foreground", "left side"]
    elif pos == "foreground_right":
        tags += ["foreground", "right side"]
    elif pos == "midground_center":
        tags += ["midground", "center composition"]

    if orient == "side_view":
        tags.append("side view")
    elif orient == "back_view":
        tags.append("from behind")
    elif orient == "front_view":
        tags.append("front view")

    if bg == "far_back":
        tags.append("deep background")
    elif bg == "far_below":
        tags += ["deep background", "view from above"]

    if layout == "layered_depth":
        tags += ["layered composition", "depth separation"]
    elif layout == "balanced_layers":
        tags += ["balanced composition", "depth separation"]
    elif layout == "negative_space":
        tags += ["negative space", "clear separation"]

    if gaze == "toward_depth":
        tags.append("looking into distance")

    if family == "urban":
        tags += ["cityscape as main subject", "urban environment dominant"]
    elif family == "indoor":
        tags += ["interior space as main subject"]
    elif family == "waterside":
        tags += ["riverside as main subject"]
    elif family == "forest":
        tags += ["forest as main subject"]
    elif family == "mountain":
        tags += ["mountain landscape as main subject"]
    elif family == "open_nature":
        tags += ["open landscape as main subject"]

    return clamp_tags(tags, 18)



def build_fixed_skeleton_tags(facts: Dict[str, Any], composition: Dict[str, Any]) -> List[str]:
    subject_hint = facts.get("subject_hint") or {}
    subject_type = str(subject_hint.get("subject_type") or "").strip().lower()
    env_tokens = build_environment_tokens(facts)

    tags = [
        "masterpiece",
        "best quality",
        "anime style",
        "manga style",
        "2d illustration",
        "wide shot",
    ]

    # 既存寄り互換: monochrome 系はデフォルト維持
    tags += ["monochrome", "manga screentone", "inked lineart", "high contrast"]
    tags += composition_to_tags(composition, facts)
    tags += env_tokens["common"]
    tags += env_tokens["subject"]
    tags += env_tokens["foreground"] + env_tokens["midground"] + env_tokens["background"]

    if subject_type == "human":
        tags += ["1 person", "small subject in frame"]
    elif subject_type == "animal":
        tags += ["small animal silhouette", "small subject in frame"]

    return clamp_tags(tags, 28)



def clean_effect_tags(tags: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in ensure_list(tags):
        n = normalize_tag(t)
        if not n:
            continue
        if len(n) > 40:
            continue
        cleaned.append(n)
    return clamp_tags(cleaned, 6)



def clean_seed_tags(tags: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in ensure_list(tags):
        n = normalize_tag(t)
        if not n:
            continue
        if any(bad in n for bad in ["beautiful", "amazing", "awesome", "perfect prompt"]):
            continue
        cleaned.append(n)
    return clamp_tags(cleaned, 10)



def resolve_position_conflict(tags: List[str], composition: Dict[str, Any]) -> List[str]:
    pos = str(composition.get("subject_position") or "")
    normalized = [normalize_tag(x) for x in ensure_list(tags) if normalize_tag(x)]
    if pos == "foreground_left":
        normalized = [t for t in normalized if t != "right side"]
    elif pos == "foreground_right":
        normalized = [t for t in normalized if t != "left side"]
    return clamp_tags(normalized, 40)



def merge_core_tags(compact_tags: List[str], must_keep: List[str], extras: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for tag in compact_tags + must_keep + extras:
        n = normalize_tag(tag)
        if n and n not in seen:
            merged.append(n)
            seen.add(n)
    return clamp_tags(merged, 32)



def compact_with_llm(
    skeleton_tags: List[str],
    effect_tags: List[str],
    seed_tags: List[str],
    composition_tags: List[str],
    composition: Dict[str, Any],
    facts: Dict[str, Any],
    llm_compact_seed: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    info = {"enabled": ENV_LLM_ENABLED, "used": False, "status": "disabled", "model": ENV_LLM_MODEL, "reason": ""}

    effect_tags = clean_effect_tags(effect_tags)
    seed_tags = clean_seed_tags(seed_tags)
    composition_tags = clamp_tags(composition_tags, 18)
    llm_compact_seed = clamp_tags(llm_compact_seed, 14)

    must_keep = ["wide shot", "anime style", "manga style", "2d illustration"]
    must_keep += [t for t in composition_tags if t in {
        "high angle", "eye level", "low angle",
        "foreground", "left side", "right side", "midground", "center composition",
        "layered composition", "depth separation", "negative space", "clear separation",
        "looking into distance",
        "cityscape as main subject", "urban environment dominant",
        "interior space as main subject", "riverside as main subject",
        "forest as main subject", "mountain landscape as main subject", "open landscape as main subject",
    }]

    fallback = merge_core_tags(llm_compact_seed, must_keep, skeleton_tags + effect_tags + seed_tags)
    fallback = resolve_position_conflict(fallback, composition)

    if not ENV_LLM_ENABLED:
        return fallback, info

    try:
        system = (
            "You are an Animagine XL 4.0 prompt optimizer. Output only comma-separated English tags. "
            "Keep environment fidelity. Do not rewrite urban into nature or indoor into outdoor. "
            "Keep composition-critical tags."
        )
        user = (
            f"environment_family: {facts.get('environment_family')}\n"
            f"primary_space: {facts.get('primary_space')}\n"
            f"scene_summary: {facts.get('scene_summary')}\n"
            f"skeleton_tags: {', '.join(skeleton_tags)}\n"
            f"effect_tags: {', '.join(effect_tags)}\n"
            f"seed_tags: {', '.join(seed_tags)}\n"
            f"composition_tags: {', '.join(composition_tags)}\n"
            f"compact_seed: {', '.join(llm_compact_seed)}\n"
            "Return the best compact final tags for Animagine."
        )
        raw = llm_generate(system, user, max_new_tokens=100)
        compact_tags = clamp_tags(split_tags(raw), 24)
        if not compact_tags:
            raise ValueError("empty compact tags")
        final_tags = merge_core_tags(compact_tags, must_keep, skeleton_tags + effect_tags + seed_tags)
        final_tags = resolve_position_conflict(final_tags, composition)
        info.update({"used": True, "status": "success"})
        return final_tags, info
    except Exception as e:
        info.update({"status": "failed", "reason": str(e)})
        return fallback, info


# =========================================================
# output
# =========================================================
def infer_focus_priority(facts: Dict[str, Any]) -> str:
    if str(facts.get("environment_family") or "") in {"urban", "indoor", "waterside", "forest", "mountain", "open_nature"}:
        return "environment"
    return "balanced"



def infer_scene_match(facts: Dict[str, Any]) -> str:
    primary = str(facts.get("primary_space") or "")
    family = str(facts.get("environment_family") or "")
    if primary:
        return primary.replace("_street", "").replace("_room", "")
    return family



def aggregate_llm_status(parts: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    used = any(bool(p.get("used")) for p in parts)
    failed = [p for p in parts if p.get("status") == "failed"]
    if failed:
        return used, "partial", "; ".join(f"{p.get('reason', '')}" for p in failed if p.get("reason"))
    if used:
        return used, "success", "design/title/compact refined with local llm"
    return False, "disabled", ""



def build_output(audio_id: str, facts: Dict[str, Any], final_tags: List[str], composition: Dict[str, Any], title: str, llm_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    onomatopoeia = facts.get("onomatopoeia") or {}
    panel_plan = [{
        "panel_index": 1,
        "shot": "wide",
        "angle": str(composition.get("subject_orientation") or "side_view"),
        "composition": "wide_shot",
        "subject_present": True,
        "subject_scale": "small",
        "focus_priority": infer_focus_priority(facts),
    }]
    ono_layout = []
    if onomatopoeia.get("primary"):
        ono_layout = [{
            "text": str(onomatopoeia.get("primary") or ""),
            "mode": "overlay",
            "anchor": "diagonal",
            "intensity": str(onomatopoeia.get("intensity") or "medium"),
        }]

    llm_used, llm_status, llm_reason = aggregate_llm_status(llm_parts)

    return {
        "audio_id": audio_id,
        "title": title,
        "manga_title": title,
        "scene_match": infer_scene_match(facts),
        "confidence": 0.8,
        "panel_count": 1,
        "panel_plan": panel_plan,
        "positive_prompt": ", ".join(final_tags),
        "negative_prompt": NEGATIVE_PROMPT,
        "onomatopoeia_layout": ono_layout,
        "direction_notes": [
            "pipeline-compatible generic prompt planner",
            "llm-guided design with deterministic fallback",
            "onomatopoeia should be overlaid after image generation",
        ],
        "llm": {
            "enabled": ENV_LLM_ENABLED,
            "used": llm_used,
            "status": llm_status,
            "model": ENV_LLM_MODEL,
            "reason": llm_reason,
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
    design, design_llm = refine_design_with_llm(facts)

    # title は design 優先、失敗時の品質担保のため専用 title llm で再調整
    title, title_llm = refine_title_with_llm({**facts, "scene_title": design.get("title") or facts.get("scene_title")})
    facts["refined_title"] = title

    composition = design.get("composition") or build_default_composition(facts)
    effect_tags = clean_effect_tags(design.get("effect_tags") or [])
    seed_tags = clean_seed_tags(design.get("seed_tags") or [])
    llm_compact_seed = clamp_tags(design.get("compact_tags") or [], 14)

    composition_tags = composition_to_tags(composition, facts)
    skeleton_tags = build_fixed_skeleton_tags(facts, composition)
    final_tags, compact_llm = compact_with_llm(
        skeleton_tags,
        effect_tags,
        seed_tags,
        composition_tags,
        composition,
        facts,
        llm_compact_seed,
    )

    output = build_output(audio_id, facts, final_tags, composition, title, [design_llm, title_llm, compact_llm])
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
        "design_llm": design_llm,
        "title_llm": title_llm,
        "compact_llm": compact_llm,
        "design_output": design,
    }

    save_json(result_dir / OUTPUT_JSON_NAME, output)
    save_json(result_dir / DEBUG_DESIGN_NAME, debug_payload)

    log(f"Saved -> {result_dir / OUTPUT_JSON_NAME}")
    log(f"Saved -> {result_dir / DEBUG_DESIGN_NAME}")
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
