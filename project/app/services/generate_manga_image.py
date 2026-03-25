#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_manga_image_llm_v2.py

役割:
- 08_manga_prompt.json は「設計用プロンプト」とみなし、そのまま描画には使わない
- 07_scene_interpretation.json / 08_manga_prompt.json を材料に、LLM に
  「Animagine XL 4.0 で絵に効く描画用プロンプト」へ変換させる
- 変換後のプロンプトで画像生成し、既存の 09_manga_image.json 互換フォーマットで保存する

改善点:
- LLM に「抽象語は禁止、具体タグへ具体化」のルールを強化
- 変換後プロンプトの最終整形を追加（抽象語の除去、具体タグの補完）
- Animagine 向けに landscape 構図を描きやすい生成パラメータへ調整
- 入力 / 出力ファイル名と JSON の外形は変更しない
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import psutil
except ImportError:
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_MODEL_ID = "cagliostrolab/animagine-xl-4.0"
DEFAULT_LLM_MODEL = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

INPUT_PROMPT_JSON = "08_manga_prompt.json"
INPUT_SCENE_JSON = "07_scene_interpretation.json"
OUTPUT_IMAGE_NAME = "09_manga_image.png"
OUTPUT_META_NAME = "09_manga_image.json"

ENV_LLM_ENABLED = os.getenv("IMAGE_PROMPT_LLM", "1").strip().lower() in ("1", "true", "yes", "on")
ENV_LLM_MODEL = os.getenv("IMAGE_PROMPT_LLM_MODEL", DEFAULT_LLM_MODEL)
ENV_MAX_NEW_TOKENS = int(os.getenv("IMAGE_PROMPT_MAX_NEW_TOKENS", "260"))

DEFAULT_WIDTH = 1216
DEFAULT_HEIGHT = 832
DEFAULT_STEPS = 32
DEFAULT_CFG = 5.8

_tokenizer = None
_model = None


def log(msg: str) -> None:
    print(f"[generate_manga_image_llm_v2] {msg}", flush=True)


def log_memory(label: str) -> None:
    if psutil:
        mem = psutil.virtual_memory()
        log(f"[MEMORY][{label}] Used: {mem.used / (1024**3):.2f}GB / Total: {mem.total / (1024**3):.2f}GB")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any], pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)


def normalize_tag(tag: str) -> str:
    s = str(tag or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_list(x: Any):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def split_tags(text: str):
    return [normalize_tag(t) for t in re.split(r"[,\n;/]+", str(text or "")) if normalize_tag(t)]


def unique_tags(tags):
    seen = set()
    out = []
    for t in tags:
        n = normalize_tag(t)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def clean_final_prompt(prompt: str) -> str:
    return ", ".join(unique_tags(split_tags(prompt)))


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


def load_llm() -> None:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    log_memory("BEFORE_LLM_LOAD")
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

    log_memory("AFTER_LLM_LOAD")


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
        raise ValueError("empty llm response")

    patterns = [
        r"BEGIN_JSON\s*(\{.*?\})\s*END_JSON",
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))

    start = text.find("{")
    if start == -1:
        raise ValueError("json object not found")

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

    raise ValueError("valid json block not found")


ABSTRACT_BANNED = {
    "negative space",
    "clear separation",
    "empty midground space",
    "medium negative space",
    "city below",
    "low horizon city",
    "looking into distance",
    "foreground",
    "left side",
    "right side",
    "side silhouette",
    "city",
    "hill",
    "sky",
    "skyline",
}


def build_conversion_system_prompt() -> str:
    return (
        "あなたは画像生成プロンプト変換AIです。\n"
        "08_manga_prompt.json の positive_prompt は『設計図』であり、そのまま描画に使うと壊れることがあります。\n"
        "あなたの仕事は、Animagine XL 4.0 で安定して絵に効く『描画用プロンプト』へ変換することです。\n"
        "重要な法則:\n"
        "1. 抽象語は禁止。必ず具体タグへ具体化する\n"
        "   - hill → sloped grassy hill, detailed ground texture\n"
        "   - sky → dramatic sky, soft gradient sky\n"
        "   - city → thin distant skyline, far city lights, visible city silhouette\n"
        "2. 何もない真っ白画像にしない。地面・空・遠景の少なくとも3要素は描かせる\n"
        "3. 主役は小さく保つが、消さない。distant figure, small subject, standing alone を優先してよい\n"
        "4. 都市は支配的にしない。遠景の気配として扱う\n"
        "5. 鳥や影は空間スケールと物語性の補助として使う\n"
        "6. 今回は japanese manga ではなく cinematic illustration を優先する\n"
        "7. positive_prompt は英語のカンマ区切りタグのみ。単語レベルの抽象タグは禁止\n"
        "8. negative_prompt も英語のカンマ区切りタグのみ\n"
        "9. 返答は必ず BEGIN_JSON で始め、END_JSON で終えること\n"
        "10. JSON以外の説明文は禁止\n"
        "11. 返すキーは positive_prompt, negative_prompt, rationale の3つのみ\n"
    )


def build_conversion_user_prompt(scene: Dict[str, Any], prompt_data: Dict[str, Any]) -> str:
    compact = {
        "title": prompt_data.get("title"),
        "scene_summary": scene.get("scene_summary"),
        "mood_tags": ensure_list(scene.get("mood_tags"))[:6],
        "visual_hints": ensure_list(scene.get("visual_hints"))[:6],
        "audio_event_tags": ensure_list(scene.get("audio_event_tags"))[:8],
        "subject_hint": scene.get("subject_hint"),
        "narrative_hook": scene.get("narrative_hook"),
        "prompt_seed_en": ensure_list((scene.get("manga_prompt_bridge") or {}).get("prompt_seed_en"))[:12],
        "design_positive_prompt": prompt_data.get("positive_prompt", ""),
        "design_negative_prompt": prompt_data.get("negative_prompt", ""),
    }
    return (
        f"input:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "出力要件:\n"
        "- positive_prompt は描画に効く具体タグだけ\n"
        "- slope / hill / sky / skyline / birds / shadow は必ず具体化する\n"
        "- wide shot, cinematic illustration, distant figure は残してよい\n"
        "- 鳥は small birds in sky, flying birds のように具体化してよい\n"
        "- 都市は thin distant skyline, visible city silhouette, far city lights のように遠景化する\n"
        "- 作品性が必要なら looking into distance, searching, waiting, distant presence を追加してよい\n\n"
        "出力テンプレート:\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "positive_prompt": "english, comma, separated, concrete tags",\n'
        '  "negative_prompt": "english, comma, separated, tags",\n'
        '  "rationale": "日本語1文"\n'
        "}\n"
        "END_JSON"
    )


def postprocess_positive(tags):
    tags = unique_tags(tags)

    replacements = {
        "slope": ["sloped grassy hill", "detailed ground texture"],
        "hill": ["sloped grassy hill", "natural terrain"],
        "sky": ["dramatic sky", "soft gradient sky"],
        "skyline": ["thin distant skyline"],
        "cityscape": ["visible city silhouette", "far city lights"],
        "birds": ["small birds in sky", "flying birds"],
        "shadow": ["long shadow"],
    }

    final = []
    for t in tags:
        if t in ABSTRACT_BANNED:
            continue
        if t in replacements:
            final.extend(replacements[t])
        else:
            final.append(t)

    must_have = [
        "masterpiece",
        "best quality",
        "ultra detailed",
        "anime style",
        "cinematic illustration",
        "2d illustration",
        "wide shot",
        "distant figure",
        "small subject",
        "standing alone",
        "sloped grassy hill",
        "detailed ground texture",
        "open sky",
        "soft gradient sky",
        "atmospheric perspective",
        "depth",
        "cinematic composition",
    ]
    for t in must_have:
        if t not in final:
            final.append(t)

    return clean_final_prompt(", ".join(final))


def build_fallback_positive(prompt_data: Dict[str, Any], scene: Dict[str, Any]) -> str:
    raw_positive = str(prompt_data.get("positive_prompt") or "").lower()
    has_city = ("city" in raw_positive) or ("urban" in raw_positive) or ("skyline" in raw_positive)
    has_animal = ("animal" in raw_positive) or ("cat" in raw_positive)
    title = str(prompt_data.get("title") or "")

    tags = [
        "masterpiece",
        "best quality",
        "ultra detailed",
        "anime style",
        "cinematic illustration",
        "2d illustration",
        "wide shot",
        "high angle",
        "distant figure",
        "small subject",
        "standing alone",
        "sloped grassy hill",
        "detailed ground texture",
        "natural terrain",
        "open sky",
        "soft gradient sky",
        "thin distant skyline",
        "visible city silhouette",
        "far city lights",
        "atmospheric perspective",
        "depth",
        "cinematic composition",
        "quiet",
        "lonely atmosphere",
    ]

    if has_animal:
        tags += ["small animal silhouette", "long shadow"]
    else:
        tags += ["silhouette", "long shadow"]

    if has_city:
        tags += ["subtle urban edge"]

    tags += ["small birds in sky", "flying birds"]

    if "不気味" in ensure_list(scene.get("mood_tags")):
        tags += ["eerie"]
    if "不穏" in ensure_list(scene.get("mood_tags")):
        tags += ["subtle tension"]
    if "ささやき" in title:
        tags += ["hushed atmosphere"]

    return clean_final_prompt(", ".join(tags))


def build_fallback_negative(prompt_data: Dict[str, Any]) -> str:
    base = [
        "text",
        "watermark",
        "logo",
        "signature",
        "blurry",
        "bad anatomy",
        "bad hands",
        "extra fingers",
        "extra limbs",
        "deformed",
        "cropped",
        "multiple people",
        "duo",
        "crowd",
        "close-up",
        "portrait",
        "overcrowded city",
        "giant buildings",
        "detailed city",
        "busy background",
        "futuristic",
        "sci-fi",
        "abstract",
        "minimal background",
        "blank background",
        "empty image",
        "low contrast",
        "washed out",
        "vibrant colors",
        "colorful",
    ]
    raw = [normalize_tag(t) for t in str(prompt_data.get("negative_prompt") or "").split(",") if normalize_tag(t)]
    return clean_final_prompt(", ".join(base + raw))


def convert_prompt_with_llm(scene: Dict[str, Any], prompt_data: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    llm_info = {
        "enabled": ENV_LLM_ENABLED,
        "used": False,
        "status": "disabled",
        "model": ENV_LLM_MODEL,
        "reason": "",
        "raw_preview": "",
    }

    fallback_positive = build_fallback_positive(prompt_data, scene)
    fallback_negative = build_fallback_negative(prompt_data)

    if not ENV_LLM_ENABLED:
        return fallback_positive, fallback_negative, llm_info

    try:
        raw = llm_generate(
            build_conversion_system_prompt(),
            build_conversion_user_prompt(scene, prompt_data),
            max_new_tokens=200,
        )
        llm_info["raw_preview"] = raw[:1200]
        obj = extract_json_block(raw)

        positive = postprocess_positive(split_tags(obj.get("positive_prompt", "")))
        negative = clean_final_prompt(obj.get("negative_prompt", ""))

        if not positive:
            raise ValueError("empty positive_prompt")
        if not negative:
            raise ValueError("empty negative_prompt")

        llm_info.update({"used": True, "status": "success", "reason": str(obj.get("rationale") or "")})
        return positive, negative, llm_info
    except Exception as e:
        llm_info.update({"status": "failed", "reason": str(e)})
        return fallback_positive, fallback_negative, llm_info


class MangaImageGenerator:
    def __init__(self, model_id: str):
        log_memory("BEFORE_MODEL_LOAD")
        log(f"Initializing pipeline with model: {model_id}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        else:
            self.pipe.to(self.device)

        log_memory("AFTER_MODEL_LOAD")

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG,
        seed: Optional[int] = None,
    ) -> Tuple[Any, int, float, str]:
        final_pos = clean_final_prompt(prompt)
        final_neg = clean_final_prompt(negative_prompt)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        log(f"Generating image (Seed: {seed}, Size: {width}x{height})")
        log(f"Final Prompt: {final_pos[:250]}...")
        log_memory("START_GENERATE")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()

        with torch.inference_mode():
            image = self.pipe(
                prompt=final_pos,
                negative_prompt=final_neg,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]

        elapsed = time.time() - start_time
        log(f"Generation completed in {elapsed:.2f} seconds.")

        return image, seed, elapsed, final_pos

    def cleanup(self):
        log("Performing deep memory cleanup...")
        if hasattr(self, 'pipe'):
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe

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
        log_memory("AFTER_CLEANUP")


def run_generate_manga_image(
    audio_id: str,
    model_id: str = DEFAULT_MODEL_ID,
    seed: Optional[int] = None,
    pretty: bool = True
) -> Dict[str, Any]:

    result_dir = RESULTS_DIR / audio_id
    prompt_path = result_dir / INPUT_PROMPT_JSON
    scene_path = result_dir / INPUT_SCENE_JSON

    if not prompt_path.exists():
        raise FileNotFoundError(f"Input prompt JSON not found: {prompt_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene interpretation JSON not found: {scene_path}")

    prompt_data = load_json(prompt_path)
    scene_data = load_json(scene_path)

    log(f"Loaded prompts for: {prompt_data.get('title', 'Unknown')}")

    converted_positive, converted_negative, prompt_llm = convert_prompt_with_llm(scene_data, prompt_data)

    generator = MangaImageGenerator(model_id)

    try:
        image, used_seed, elapsed, cleaned_prompt = generator.generate(
            prompt=converted_positive,
            negative_prompt=converted_negative,
            seed=seed
        )

        output_image_path = result_dir / OUTPUT_IMAGE_NAME
        image.save(output_image_path)
        log(f"Saved image to: {output_image_path}")

        output_meta_path = result_dir / OUTPUT_META_NAME
        meta = {
            "audio_id": audio_id,
            "image_generation_version": "llm_illustration_animagine_v2",
            "image_file": str(OUTPUT_IMAGE_NAME),
            "model_id": model_id,
            "device": str(generator.device),
            "seed": used_seed,
            "params": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "steps": DEFAULT_STEPS,
                "cfg_scale": DEFAULT_CFG,
                "sampler": "Euler a"
            },
            "elapsed_sec": round(elapsed, 2),
            "prompt_summary": {
                "title": prompt_data.get("title", ""),
                "positive_original": clean_final_prompt(prompt_data.get("positive_prompt", "")),
                "positive_cleaned": cleaned_prompt,
                "negative_cleaned": converted_negative
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        save_json(output_meta_path, meta, pretty=pretty)
        log(f"Saved metadata to: {output_meta_path}")
    finally:
        generator.cleanup()
        del generator

        # --- ここから追加 ---
        global _model, _tokenizer
        _model = None
        _tokenizer = None

        import gc
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
        # --- ここまで追加 ---

        log("Process resources cleared.")

    return meta


def run(audio_id: str, pretty: bool = True):
    return run_generate_manga_image(audio_id=audio_id, pretty=pretty)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-refined illustration image")
    parser.add_argument("--audio-id", required=True, help="Processing Audio ID")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HuggingFace Model ID")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed")
    parser.add_argument("--pretty", action="store_true", help="Format JSON")
    args = parser.parse_args()

    try:
        result = run_generate_manga_image(
            audio_id=args.audio_id,
            model_id=args.model,
            seed=args.seed,
            pretty=args.pretty
        )
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    except Exception as e:
        log(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
