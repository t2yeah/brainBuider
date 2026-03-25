
1. 先頭のパス設定を相対化
今のコード
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_onomatopoeia.py

役割:
- 物理特徴、オーディオイベント、空間判定、シーン解釈を統合。
- LLM (Swallow-8B) を用いて、主役の行動にフォーカスしたオノマトペを生成。
- 日本の漫画表現（オギャー、ギャー、アハハ等）として自然な擬音を最優先する。
- 異常な出力（タイポ、スペース、全角記号、解説文）に対する「超」堅牢なパース機能を備える。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def download_lora_from_hf(repo_id: str) -> str:
    from huggingface_hub import snapshot_download
    project_root = Path(__file__).resolve().parents[2]
    HF_CACHE_DIR = project_root / "models" / "hf"
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    """
    HuggingFaceからLoRAをダウンロードしてローカルに保存
    """
    local_dir = HF_CACHE_DIR / repo_id

    if local_dir.exists():
        log(f"LoRA already cached: {local_dir}")
        return str(local_dir)

    log(f"Downloading LoRA from HF: {repo_id}")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    log(f"Downloaded LoRA to: {path}")
    return path


# =========================================================
# パス・定数設定
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_LLM_MODEL = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
ENV_LLM_ENABLED = os.getenv("MANGA_PROMPT_LLM", "1").strip().lower() in ("1", "true", "yes", "on")
ENV_LLM_MODEL = os.getenv("MANGA_PROMPT_LLM_MODEL", DEFAULT_LLM_MODEL)

INPUT_FILES = [
    "04_features.json",
    "05_audio_events.json",
    "06_space_judgement.json",
    "07_scene_interpretation.json",
]

OUTPUT_JSON_NAME = "08_onomatopoeia.json"
# LoRA adapter
DEFAULT_LORA_ADAPTER_DIR = str(
    PROJECT_ROOT / "train" / "onomato-30000" / "outputs" / "onoma-lora-swallow-v2" / "final_adapter"
)
DEFAULT_LORA_REPO = "yadorigi/onomatopoeia-lora"
ENV_LORA_REPO = os.getenv("ONOMA_LORA_REPO", DEFAULT_LORA_REPO)
# =========================================================
# ユーティリティ
# =========================================================
def log(msg: str):
    print(f"[agent_onomatopoeia] {msg}", file=sys.stderr, flush=True)

def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists(): return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return None

def estimate_intensity(features: Dict) -> str:
    """音の強さを簡易的に判定"""
    rms = float(features.get("rms_mean", 0.0))
    if rms >= 0.05: return "strong"
    if rms >= 0.01: return "medium"
    return "soft"

def normalize_intensity_label(v: str) -> str:
    s = str(v or "").strip().lower()
    if s in ("strong", "high", "loud"):
        return "strong"
    if s in ("medium", "mid", "moderate"):
        return "medium"
    if s in ("weak", "soft", "low", "quiet"):
        return "weak"
    return "medium"


def normalize_proximity_label(v: str) -> str:
    s = str(v or "").strip().lower()
    if s in ("near", "close", "foreground"):
        return "near"
    if s in ("far", "distant", "background"):
        return "far"
    return "mid"


def classify_family_from_event(event_label: str) -> str:
    e = str(event_label or "").strip().lower()

    if any(k in e for k in ["water", "splash", "drip", "rain", "stream", "river", "wave"]):
        return "water"
    if any(k in e for k in ["leaf", "forest", "bird", "branch", "underbrush"]):
        return "forest"
    if any(k in e for k in ["footstep", "step", "walk", "sneak"]):
        return "footstep"
    if any(k in e for k in ["silence", "stillness", "ominous", "tension"]):
        return "silence_tension"
    if any(k in e for k in ["wind", "gust", "whirr", "whoosh"]):
        return "wind"
    if any(k in e for k in ["buzz", "hum", "fan", "motor", "machine", "electric"]):
        return "machine"
    if any(k in e for k in ["metal", "clink", "clang", "impact", "crash", "break"]):
        return "impact"
    if any(k in e for k in ["baby", "cry", "laugh", "scream", "voice"]):
        return "voice"
    if any(k in e for k in ["cat", "dog", "frog", "animal"]):
        return "animal"
    return "environment"


def infer_mood(scene_title: str, event_labels: List[str]) -> str:
    text = " ".join([str(scene_title or "")] + [str(x) for x in event_labels]).lower()

    if any(k in text for k in ["horror", "ominous", "abandoned", "scary", "恐怖", "不気味", "廃屋", "気配"]):
        return "horror"
    if any(k in text for k in ["tense", "tension", "sneak", "緊張", "忍び", "静まり"]):
        return "tense"
    if any(k in text for k in ["mysterious", "mystery", "mysterious", "神秘", "不思議", "森"]):
        return "mysterious"
    if any(k in text for k in ["fun", "happy", "laugh", "明るい", "楽しい"]):
        return "bright"
    return "neutral"


def extract_primary_event(ae: Dict[str, Any]) -> str:
    events = ae.get("global_top_events", []) if ae else []
    if events:
        top = events[0]
        if isinstance(top, dict):
            return str(top.get("label") or "unknown_event")
    return "unknown_event"


def extract_space_label(sj: Dict[str, Any]) -> str:
    return str((sj or {}).get("final_space_label") or "mixed_ambiguous")


def build_cond_context(
    features_json: Dict[str, Any],
    ae: Dict[str, Any],
    sj: Dict[str, Any],
    si: Dict[str, Any],
) -> Dict[str, Any]:
    scene_title = str((si or {}).get("scene_title") or "不明なシーン")
    event_labels = [str(e.get("label")) for e in (ae or {}).get("global_top_events", []) if isinstance(e, dict)]
    event_strengths = extract_global_event_strengths(ae)

    primary_event = extract_primary_event(ae)
    family = classify_family_from_event(primary_event)
    space = extract_space_label(sj)

    first_seg_features = ((features_json or {}).get("segment_features") or [{}])[0].get("features", {})
    intensity = estimate_intensity(first_seg_features)
    intensity = normalize_intensity_label(intensity)

    mood = infer_mood(scene_title, event_labels)
    proximity = "mid"

    speech = float(event_strengths.get("Speech", 0.0))
    vehicle = float(event_strengths.get("Vehicle", 0.0)) + float(event_strengths.get("Car", 0.0)) + float(event_strengths.get("Bus", 0.0))
    music = float(event_strengths.get("Music", 0.0))

    if space == "urban_street" and speech >= 0.35 and (vehicle >= 0.18 or music >= 0.12):
        family = "environment"
        mood = "busy"

    context = {
        "scene_title": scene_title,
        "audio_events": event_labels,
        "event_strengths": event_strengths,
        "event": primary_event,
        "family": family,
        "space": space,
        "intensity": intensity,
        "mood": mood,
        "proximity": proximity,
    }

    mode = decide_onomato_mode(context)
    context["mode"] = mode

    cond = (
        f"mode={mode}; "
        f"event={primary_event}; "
        f"family={family}; "
        f"space={space}; "
        f"intensity={intensity}; "
        f"mood={mood}; "
        f"proximity={proximity}"
    )
    context["cond"] = cond
    context["candidate_pool"] = choose_candidate_pool_by_mode(context)

    return context

def extract_global_event_strengths(ae: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ev in (ae or {}).get("global_top_events", []):
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("label") or "").strip()
        if not label:
            continue
        try:
            score = float(ev.get("mean_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        out[label] = score
    return out

def decide_onomato_mode(context: Dict[str, Any]) -> str:
    strengths = context.get("event_strengths", {}) or {}
    space = str(context.get("space") or "")

    speech = float(strengths.get("Speech", 0.0))
    vehicle = (
        float(strengths.get("Vehicle", 0.0))
        + float(strengths.get("Car", 0.0))
        + float(strengths.get("Bus", 0.0))
    )
    music = float(strengths.get("Music", 0.0))
    footsteps = float(strengths.get("Footsteps", 0.0)) + float(strengths.get("Shuffle", 0.0))
    water = float(strengths.get("Waves, surf", 0.0)) + float(strengths.get("Water", 0.0))
    wind = float(strengths.get("Wind noise (microphone)", 0.0)) + float(strengths.get("Wind", 0.0))

    primary_event = str(context.get("event") or "").lower()
    family = str(context.get("family") or "").lower()

    complex_env = speech + vehicle + music + footsteps

    # 都市雑踏・駅前・店先・歩道などの複合環境音
    if space == "urban_street" and complex_env >= 0.75:
        return "ambience"

    if speech >= 0.40 and (vehicle >= 0.18 or music >= 0.12):
        return "ambience"

    if footsteps >= 0.08 and speech >= 0.20:
        return "ambience"

    # 明確な単音イベント寄り
    if family in {"water", "impact", "machine", "animal", "voice"}:
        if water >= 0.25 and water > speech and water > vehicle:
            return "single_event"
        if wind >= 0.20 and wind > speech and wind > vehicle:
            return "single_event"
        if primary_event not in {"speech", "vehicle", "music", "shuffle"}:
            return "single_event"

    return "single_event"


def choose_single_event_candidates(context: Dict[str, Any]) -> List[str]:
    strengths = context.get("event_strengths", {}) or {}
    event = str(context.get("event") or "")
    family = str(context.get("family") or "")
    scene_title = str(context.get("scene_title") or "")

    speech = float(strengths.get("Speech", 0.0))
    vehicle = float(strengths.get("Vehicle", 0.0)) + float(strengths.get("Car", 0.0)) + float(strengths.get("Bus", 0.0))
    water = float(strengths.get("Waves, surf", 0.0)) + float(strengths.get("Water", 0.0))
    wind = float(strengths.get("Wind noise (microphone)", 0.0)) + float(strengths.get("Wind", 0.0))

    e = event.lower()
    f = family.lower()
    t = scene_title.lower()

    if "baby" in e or "cry" in e or any(x in t for x in ["赤ちゃん", "泣"]):
        return ["オギャー！", "エーン", "ワーン", "ヒック", "メソメソ"]

    if "cat" in e:
        return ["ニャー", "ミャー", "ニャン", "シャー", "ゴロゴロ"]

    if "dog" in e:
        return ["ワンワン", "キャンキャン", "クゥーン", "ガルル", "ハッハッ"]

    if f == "water" or water >= 0.20:
        return ["ザブーン", "チャプチャプ", "ザザー", "サーッ", "バシャッ"]

    if f == "wind" or wind >= 0.18:
        return ["ビュウ", "ヒュウ", "ゴォ", "サァー", "ザワ…"]

    if f == "machine":
        return ["ウィーン", "ブーン", "ガコン", "キュイーン", "ジジジ"]

    if f == "impact":
        return ["バン", "ドン", "ガシャ", "ガツン", "ドゴッ"]

    if f == "voice":
        if speech >= 0.30 and vehicle < 0.10:
            return ["ヒソヒソ", "ヒソ…", "ボソボソ", "コソコソ", "ザワ…"]
        return ["ワッ", "アッ", "ウワッ", "ギャッ", "キャー"]

    return ["ドン", "バン", "ガサッ", "コツ", "……"]


def choose_ambience_candidates(context: Dict[str, Any]) -> List[str]:
    strengths = context.get("event_strengths", {}) or {}
    space = str(context.get("space") or "")
    scene_title = str(context.get("scene_title") or "")

    speech = float(strengths.get("Speech", 0.0))
    vehicle = float(strengths.get("Vehicle", 0.0)) + float(strengths.get("Car", 0.0)) + float(strengths.get("Bus", 0.0))
    music = float(strengths.get("Music", 0.0))
    footsteps = float(strengths.get("Footsteps", 0.0)) + float(strengths.get("Shuffle", 0.0))
    water = float(strengths.get("Waves, surf", 0.0)) + float(strengths.get("Water", 0.0))
    wind = float(strengths.get("Wind noise (microphone)", 0.0)) + float(strengths.get("Wind", 0.0))

    title_l = scene_title.lower()

    if space == "urban_street":
        if speech >= 0.40 and (vehicle >= 0.20 or music >= 0.15):
            return ["ザワザワ", "ガヤガヤ", "ワイワイ", "ゴー", "コツコツ"]
        if footsteps >= 0.04 and speech >= 0.25:
            return ["コツコツ", "カツカツ", "ザッザッ", "ザワザワ", "ガヤガヤ"]
        if vehicle >= 0.30:
            return ["ゴー", "ブーン", "ザワザワ", "コツコツ", "ガヤガヤ"]
        return ["ザワザワ", "ガヤガヤ", "コツコツ", "ゴー", "ワイワイ"]

    if space == "indoor_room":
        if speech >= 0.30 and vehicle < 0.10 and music < 0.10:
            return ["ザワ…", "ヒソヒソ", "ボソボソ", "コツコツ", "コソコソ"]
        if footsteps >= 0.05:
            return ["コツコツ", "カツカツ", "ザワ…", "ガヤガヤ", "ヒソヒソ"]
        return ["ザワ…", "コツコツ", "ヒソヒソ", "ガヤガヤ", "ボソボソ"]

    if water >= 0.18 and water >= speech and water >= vehicle:
        return ["ザザー", "サーッ", "チャプチャプ", "ザブーン", "ゴォォ"]

    if wind >= 0.15 and wind > speech and wind > vehicle:
        return ["ビュウ", "ゴォ", "サァー", "ヒュウ", "ザワ…"]

    if "楽しい" in scene_title or "賑" in scene_title or "祭" in title_l:
        return ["ワイワイ", "ガヤガヤ", "ザワザワ", "ドッ", "コツコツ"]

    return ["ザワザワ", "ガヤガヤ", "コツコツ", "ゴー", "……"]


def choose_candidate_pool_by_mode(context: Dict[str, Any]) -> List[str]:
    mode = str(context.get("mode") or "single_event")
    if mode == "ambience":
        return choose_ambience_candidates(context)
    return choose_single_event_candidates(context)

def choose_candidate_pool(context: Dict[str, Any]) -> List[str]:
    strengths = context.get("event_strengths", {}) or {}
    space = str(context.get("space") or "")
    scene_title = str(context.get("scene_title") or "")

    speech = float(strengths.get("Speech", 0.0))
    vehicle = float(strengths.get("Vehicle", 0.0)) + float(strengths.get("Car", 0.0)) + float(strengths.get("Bus", 0.0))
    music = float(strengths.get("Music", 0.0))
    footsteps = float(strengths.get("Footsteps", 0.0)) + float(strengths.get("Shuffle", 0.0))
    water = float(strengths.get("Waves, surf", 0.0)) + float(strengths.get("Water", 0.0))
    wind = float(strengths.get("Wind noise (microphone)", 0.0)) + float(strengths.get("Wind", 0.0))

    title_l = scene_title.lower()

    # 都市雑踏
    if space == "urban_street":
        if speech >= 0.40 and (vehicle >= 0.20 or music >= 0.15):
            return ["ザワザワ", "ガヤガヤ", "ワイワイ", "ゴー", "コツコツ"]
        if footsteps >= 0.04 and speech >= 0.25:
            return ["コツコツ", "カツカツ", "ザッザッ", "ザワザワ", "ガヤガヤ"]
        if vehicle >= 0.30:
            return ["ゴー", "ブーン", "ザワザワ", "コツコツ", "ガヤガヤ"]
        return ["ザワザワ", "ガヤガヤ", "コツコツ", "ゴー", "ワイワイ"]

    # 屋内で静かな会話
    if space == "indoor_room":
        if speech >= 0.30 and vehicle < 0.10 and music < 0.10:
            return ["ヒソヒソ", "ヒソ…", "ザワ…", "コソコソ", "ボソボソ"]
        return ["ザワ…", "コツコツ", "ガヤガヤ", "ヒソヒソ", "ボソボソ"]

    # 水辺
    if water >= 0.18 and water >= speech and water >= vehicle:
        return ["ザブーン", "サーッ", "チャプチャプ", "ゴォォ", "ザザー"]

    # 風
    if wind >= 0.15 and wind > speech and wind > vehicle:
        return ["ビュウ", "ゴォ", "サァー", "ヒュウ", "ザワ…"]

    # 明るい場面
    if "楽しい" in scene_title or "賑" in scene_title or "祭" in title_l:
        return ["ワイワイ", "ガヤガヤ", "ザワザワ", "ドッ", "コツコツ"]

    # デフォルト
    return ["ザワザワ", "ガヤガヤ", "コツコツ", "ゴー", "……"]


def normalize_onomato_choice(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace("　", "").replace(" ", "")
    s = s.replace("。", "").replace("、", "").replace(",", "")
    s = s.replace('"', "").replace("'", "")
    return s


def resolve_choice_from_candidates(raw_text: str, candidates: List[str]) -> Optional[str]:
    cleaned = normalize_onomato_choice(raw_text)

    if cleaned in candidates:
        return cleaned

    for c in candidates:
        if c in cleaned:
            return c

    # よくある揺れの吸収
    alias_map = {
        "ざわ…": "ザワザワ",
        "ざわざわ": "ザワザワ",
        "ガヤ…": "ガヤガヤ",
        "がやがや": "ガヤガヤ",
        "わいわい": "ワイワイ",
        "こつこつ": "コツコツ",
        "かつかつ": "カツカツ",
        "ごー": "ゴー",
        "ぶーん": "ブーン",
        "ひそ…": "ヒソ…",
        "ひそひそ": "ヒソヒソ",
        "ざぶーん": "ザブーン",
        "ちゃぷちゃぷ": "チャプチャプ",
        "びゅう": "ビュウ",
        "ごぉ": "ゴー",
        "ごぉ…": "ゴー",
        "ゴォ": "ゴー",
        "ゴォ…": "ゴー",
        "ごお": "ゴー",
        "ごお…": "ゴー",
        "ゴオ": "ゴー",
        "ゴオ…": "ゴー",
        "ざわ": "ザワザワ",
        "がや": "ガヤガヤ",
        "こつ": "コツコツ",
        "かつ": "カツカツ",
        "ざっざっ": "ザッザッ",
    }

    key = cleaned.lower()
    if key in alias_map:
        mapped = alias_map[key]
        if mapped in candidates:
            return mapped

    return None


def postprocess_primary_onomato(primary: str, context: Dict[str, Any], candidates: List[str]) -> str:
    mode = str(context.get("mode") or "single_event")
    space = str(context.get("space") or "")
    strengths = context.get("event_strengths", {}) or {}

    speech = float(strengths.get("Speech", 0.0))
    vehicle = float(strengths.get("Vehicle", 0.0)) + float(strengths.get("Car", 0.0)) + float(strengths.get("Bus", 0.0))
    music = float(strengths.get("Music", 0.0))
    water = float(strengths.get("Waves, surf", 0.0)) + float(strengths.get("Water", 0.0))

    quiet_words = {"ヒソ…", "ヒソヒソ", "コソコソ", "ボソボソ"}
    ambience_words = {"ザワザワ", "ガヤガヤ", "ワイワイ", "ゴー", "コツコツ", "カツカツ", "ザッザッ"}
    water_words = {"ザブーン", "チャプチャプ", "ザザー", "サーッ", "バシャッ"}
    wind_words = {"ビュウ", "ヒュウ", "サァー"}
    single_impact_words = {"ドン", "バン", "ガシャ", "ガツン", "ドゴッ"}

    if not primary:
        return candidates[0] if candidates else "……"

    if mode == "ambience":
        if primary in quiet_words and (speech >= 0.35 or vehicle >= 0.20 or music >= 0.15):
            return candidates[0] if candidates else "ザワザワ"
        if primary in water_words or primary in wind_words:
            return candidates[0] if candidates else "ザワザワ"
        if space == "urban_street" and primary == "ゴー" and speech >= 0.45:
            return "ザワザワ"
        return primary

    # single_event 側なのに環境擬音へ流れたら戻す
    if mode == "single_event":
        if primary in ambience_words:
            return candidates[0] if candidates else primary
        if primary == "ヒソ…" and water >= 0.20:
            return candidates[0] if candidates else "ザブーン"
        if space == "urban_street" and primary in quiet_words and vehicle >= 0.20:
            return candidates[0] if candidates else primary
        return primary

    return primary


def cleanup_generated_text(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        gen = full_text[len(prompt):]
    else:
        gen = full_text

    gen = gen.strip()

    for sep in ["\n", "。", "条件:", "出力:"]:
        if sep in gen:
            gen = gen.split(sep)[0].strip()

    for sep in ["(", "（", "{", "}", "[", "]", '"', "'"]:
        gen = gen.replace(sep, "")

    gen = re.sub(r"\s+", "", gen)

    # 先頭の擬音っぽい塊を優先して拾う
    m = re.search(r"[ぁ-んァ-ヶー…ッっ゛゜！ー]+", gen)
    if m:
        gen = m.group(0)

    return gen[:20].strip() or "……"


def load_llm_and_tokenizer(model_name: str, adapter_dir: str = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    # ★ここを変更
    repo_id = ENV_LORA_REPO
    adapter_path = download_lora_from_hf(repo_id)

    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer

def release_torch_memory(*objs):
    try:
        import gc
        import torch
    except ImportError:
        return

    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

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


# =========================================================
# 進化した洗浄・抽出ロジック (v12)
# =========================================================
def clean_text_for_json(text: str) -> str:
    """全角記号、二重中括弧、引用符を標準に変換する"""
    table = {
        '“': '"', '”': '"', '‘': "'", '’': "'", '：': ':', '，': ',', 
        '；': ';', '（': '(', '）': ')', '［': '[', '］': ']', '｛': '{', '｝': '}'
    }
    for k, v in table.items():
        text = text.replace(k, v)
    
    # 二重カッコの修正 ( {{ -> { , }} -> } )
    text = re.sub(r'^\{+', '{', text.strip())
    text = re.sub(r'\}+$', '}', text.strip())
    return text

def normalize_onomato_dict(data: Dict) -> Dict:
    """
    LLMが独自のキー名（スペース入り等）やネストした構造を作った場合に、標準形式に強制変換する
    """
    norm = {"primary": None, "candidates": [], "style": "normal", "placement": "center"}
    
    for k, v in data.items():
        # キー名のトリミングと小文字化を行い、前方一致で判定
        kl = k.strip().lower()
        if kl.startswith("prim"): norm["primary"] = v
        elif kl.startswith("cand"): norm["candidates"] = v
        elif kl.startswith("styl"): norm["style"] = v
        elif kl.startswith("plac"): norm["placement"] = v

    # candidates の平坦化 (文字列以外の要素を排除)
    if isinstance(norm["candidates"], list):
        flat_list = []
        for item in norm["candidates"]:
            if isinstance(item, dict):
                val = next(iter(item.values())) if item else None
                if val: flat_list.append(str(val))
            elif isinstance(item, str):
                flat_list.append(item)
        norm["candidates"] = flat_list

    return norm

def extract_onomato_data_v12(text: str) -> Optional[Dict]:
    """
    あらゆる異常な形式から、純粋なオノマトペ情報を抜き出す
    """
    clean_txt = clean_text_for_json(text)
    
    # 1. JSONパース試行
    try:
        match = re.search(r'(\{.*\})', clean_txt, re.DOTALL)
        if match:
            raw_json = match.group(1)
            # さらに細かく {{...}} を {...} に直す
            if raw_json.startswith("{{"): raw_json = raw_json[1:]
            if raw_json.endswith("}}"): raw_json = raw_json[:-1]
            
            data = json.loads(raw_json)
            if isinstance(data, dict):
                return normalize_onomato_dict(data)
    except:
        pass

    # 2. 正規表現スカベンジ（タイポや不完全なJSONからの救出）
    log("Regex Scavenger: Pulling strings from raw text...")
    res = {"primary": None, "candidates": [], "style": "normal", "placement": "center"}
    
    # primary: "..." (タイポやスペースを考慮)
    p_match = re.search(r'["\']\s*prim[^"\']*["\']\s*[:：]\s*["\']?([^"\'\s,，}]+)["\']?', clean_txt)
    if p_match: res["primary"] = p_match.group(1)
    
    # 全体からカタカナ語を収集
    all_quoted = re.findall(r'["\']([^"\'\s,，:：]+[ァ-ヶー！]+[^"\'\s,，:：]*)["\']', clean_txt)
    if all_quoted:
        if not res["primary"]: res["primary"] = all_quoted[0]
        res["candidates"] = list(set(all_quoted))

    return res if res["primary"] else None

# =========================================================
# LLM 生成ロジック
# =========================================================
def llm_generate_onomato_data(
    model_name: str,
    context: Dict[str, Any],
) -> Tuple[bool, Optional[Dict], str]:
    model = None
    tokenizer = None
    inputs = None
    output_ids = None

    try:
        import torch
    except ImportError:
        return False, None, "torch not found"

    try:
        model, tokenizer = load_llm_and_tokenizer(model_name)
    except Exception as e:
        return False, None, f"model load failed: {e}"

    cond = context.get("cond") or ""
    candidates = context.get("candidate_pool") or []
    if not cond:
        release_torch_memory(model, tokenizer)
        return False, None, "cond is empty"
    if not candidates:
        release_torch_memory(model, tokenizer)
        return False, None, "candidate_pool is empty"

    candidate_text = " / ".join(candidates)

    prompt = (
        "あなたは漫画向けオノマトペ生成アシスタントです。\n"
        "次の条件に最も自然な日本語オノマトペを、候補の中から1つだけ選んでください。\n"
        "候補以外は出力しないでください。\n"
        "説明文、記号、JSON、補足は禁止です。\n"
        "mode=ambience の場合は、環境全体を表す擬音を優先してください。\n"
        "mode=single_event の場合は、主役の単発音を表す擬音を優先してください。\n\n"
        f"条件:\n{cond}\n\n"
        f"候補:\n{candidate_text}\n\n"
        "出力:\n"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        log(f"Generating manga onomatopoeia with cond: {cond}")
        log(f"Candidate pool: {candidate_text}")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                top_p=None,
                temperature=None,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned = cleanup_generated_text(full_text, prompt)
        chosen = resolve_choice_from_candidates(cleaned, candidates)

        log(f"RAW GEN  : {full_text}")
        log(f"CLEAN GEN: {cleaned}")
        log(f"CHOSEN   : {chosen}")

        if not chosen:
            chosen = candidates[0]

        parsed = {
            "primary": chosen,
            "candidates": candidates[1:4],
            "style": "bold" if chosen in {"ガヤガヤ", "ワイワイ", "ゴー", "ザブーン"} else "normal",
            "placement": "center",
        }
        return True, parsed, "ok"

    except Exception as e:
        return False, None, str(e)

    finally:
        release_torch_memory(output_ids, inputs, model, tokenizer)

def build_onomato_self_check_system_prompt() -> str:
    return (
        "あなたは漫画向けオノマトペのレビュアーです。"
        "一度選ばれたオノマトペが、音イベント・空間・雰囲気に対して自然か確認してください。"
        "候補以外を新しく作ってはいけません。"
        "必ず候補の中から1つだけ選んでください。"
        "mode=ambience なら環境全体の気配を表す擬音を優先してください。"
        "mode=single_event なら主役の単発音を表す擬音を優先してください。"
        "出力は候補そのもの1語だけにしてください。説明は禁止です。"
    )


def build_onomato_self_check_user_prompt(
    context: Dict[str, Any],
    current_primary: str,
    candidates: List[str],
) -> str:
    cond = context.get("cond") or ""
    scene_title = context.get("scene_title") or ""
    event_labels = context.get("audio_events") or []
    strengths = context.get("event_strengths") or {}
    mode = context.get("mode") or "single_event"

    return (
        "以下のオノマトペ選択結果を見直してください。\n"
        "見直し観点:\n"
        "1. 他の音イベントを見落としていないか\n"
        "2. mode と擬音の種類が合っているか\n"
        "3. ambience なのに単発音すぎないか\n"
        "4. single_event なのに環境音すぎないか\n"
        "5. 候補の中で、より自然なものがあれば差し替える\n\n"
        f"scene_title:\n{scene_title}\n\n"
        f"cond:\n{cond}\n\n"
        f"mode:\n{mode}\n\n"
        f"audio_events:\n{json.dumps(event_labels, ensure_ascii=False)}\n\n"
        f"event_strengths:\n{json.dumps(strengths, ensure_ascii=False, indent=2)}\n\n"
        f"current_choice:\n{current_primary}\n\n"
        f"candidates:\n{' / '.join(candidates)}\n\n"
        "出力:\n"
    )


def llm_self_check_onomato_choice(
    model_name: str,
    context: Dict[str, Any],
    current_primary: str,
    candidates: List[str],
) -> Tuple[bool, Optional[str], str]:
    if not candidates:
        return False, None, "candidate_pool is empty"

    model = None
    tokenizer = None
    inputs = None
    output_ids = None

    try:
        import torch
    except ImportError:
        return False, None, "torch not found"

    try:
        model, tokenizer = load_llm_and_tokenizer(model_name)
    except Exception as e:
        return False, None, f"model load failed: {e}"

    system_prompt = build_onomato_self_check_system_prompt()
    user_prompt = build_onomato_self_check_user_prompt(context, current_primary, candidates)

    prompt = (
        f"<|system|>\n{system_prompt}\n\n"
        f"<|user|>\n{user_prompt}\n\n"
        f"<|assistant|>\n"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        log(f"Self-check onomatopoeia: current={current_primary}")
        log(f"Self-check candidates: {' / '.join(candidates)}")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                top_p=None,
                temperature=None,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned = cleanup_generated_text(full_text, prompt)
        chosen = resolve_choice_from_candidates(cleaned, candidates)

        log(f"SELF RAW  : {full_text}")
        log(f"SELF CLEAN: {cleaned}")
        log(f"SELF PICK : {chosen}")

        if not chosen:
            return False, None, "self-check could not resolve candidate"

        return True, chosen, "ok"

    except Exception as e:
        return False, None, str(e)

    finally:
        release_torch_memory(output_ids, inputs, model, tokenizer)


# =========================================================
# メイン処理
# =========================================================
def run(
    audio_id: str,
    llm_enabled: bool = ENV_LLM_ENABLED,
    llm_model: str = ENV_LLM_MODEL,
) -> Dict:
    result_dir = RESULTS_DIR / audio_id
    source_data = {fname: safe_read_json(result_dir / fname) for fname in INPUT_FILES}
    
    f4 = source_data.get("04_features.json") or {}
    ae = source_data.get("05_audio_events.json") or {}
    sj = source_data.get("06_space_judgement.json") or {}
    si = source_data.get("07_scene_interpretation.json") or {}

    context = build_cond_context(f4, ae, sj, si)
    event_labels = context["audio_events"]
    scene_title = context["scene_title"]

    llm_res = {"enabled": llm_enabled, "used": False, "status": "disabled", "model": llm_model, "reason": ""}
    llm_data = None

    if llm_enabled:
        ok, parsed, reason = llm_generate_onomato_data(
            llm_model,
            context,

        )

        if ok and parsed:
            llm_res.update({
                "used": True,
                "status": "success",
                "adapter_repo": ENV_LORA_REPO,
                "cond": context.get("cond", ""),
                "mode": context.get("mode", "single_event"),
            })
            llm_data = parsed

            # 2段目レビューを1回だけ実施
            base_primary = str(parsed.get("primary") or "")
            candidates = context.get("candidate_pool", []) or []

            if base_primary and candidates:
                ok2, reviewed_primary, reason2 = llm_self_check_onomato_choice(
                    llm_model,
                    context,
                    current_primary=base_primary,
                    candidates=candidates,
                )

                if ok2 and reviewed_primary:
                    llm_data["primary"] = reviewed_primary
                    llm_res["self_check"] = {
                        "used": True,
                        "status": "success",
                        "reason": reason2,
                    }
                else:
                    llm_res["self_check"] = {
                        "used": True,
                        "status": "failed",
                        "reason": reason2,
                    }

        else:
            llm_res.update({
                "status": "failed",
                "reason": reason,
                "adapter_repo": ENV_LORA_REPO,
                "cond": context.get("cond", ""),
                "mode": context.get("mode", "single_event"),
            })

    # 最終的な擬音の決定
    candidates = context.get("candidate_pool", []) or []
    primary = llm_data.get("primary") if llm_data else None
    primary = postprocess_primary_onomato(primary, context, candidates)

    # 旧・緊急避難ロジックも一部維持
    if primary:
        p_l = str(primary).lower()
        t_l = str(scene_title).lower()

        if any(x in t_l for x in ["赤ちゃん", "baby", "子供", "child", "泣"]):
            if any(x in p_l for x in ["ミー", "ミャ", "ニャ", "クークー", "ウー", "ピー", "エェン", "アーアア"]):
                log(f"Safety correction: '{primary}' -> 'オギャー！'")
                primary = "オギャー！"

        if ("ニャ" in p_l or "ミャ" in p_l) and not any(x in t_l for x in ["猫", "cat"]):
            primary = candidates[0] if candidates else "ざわ…"

    if not primary:
        primary = candidates[0] if candidates else "……"

    style_val = (llm_data.get("style") if llm_data else None) or ("bold" if "！" in str(primary) else "normal")
    placement_val = (llm_data.get("placement") if llm_data else None) or "center"

    result = {
        "audio_id": audio_id,
        "onomatopoeia_version": "v14_lora_cond_prompt_selfcheck",
        "scene_title": scene_title,
        "scene_match": sj.get("final_space_label", "mixed_ambiguous"),
        "confidence": sj.get("confidence", 0.0),
        "primary_onomatopoeia": str(primary),
        "secondary_candidates": [str(x) for x in (llm_data.get("candidates", []) if llm_data else context.get("candidate_pool", [])[1:4])],
        "intensity": estimate_intensity(f4.get("segment_features", [{}])[0].get("features", {})),
        "style_hint": str(style_val),
        "placement_hint": str(placement_val),
        "segment_onomatopoeia": [],
        "llm_info": llm_res,
        "summary": f"Generated '{primary}' from cond-based LoRA prompt for: {scene_title}"
    }

    for seg in f4.get("segment_features", []):
        result["segment_onomatopoeia"].append({
            "segment_id": seg.get("segment_id"),
            "primary_onomatopoeia": result["primary_onomatopoeia"],
            "style_hint": result["style_hint"]
        })

    out_path = result_dir / OUTPUT_JSON_NAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    log(f"Saved results (v13) to {out_path}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Agent Onomatopoeia Unit Test")
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--llm-model", default=ENV_LLM_MODEL)
    args = parser.parse_args()

    llm_enabled = ENV_LLM_ENABLED and (not args.disable_llm)

    try:
        res = run(
            args.audio_id,
            llm_enabled=llm_enabled,
            llm_model=args.llm_model,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()