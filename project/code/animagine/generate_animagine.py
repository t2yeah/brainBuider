import json
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from code.common.paths import (
    ANIMAGINE_MODEL_DIR,
    ANIMAGINE_IMAGES_DIR,
    ANIMAGINE_METADATA_DIR,
    ANIMAGINE_PROMPTS_DIR,
    ensure_animagine_dirs,
)
from code.animagine.prompt_builder import default_negative_prompt


def load_pipeline():
    if not ANIMAGINE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Animagine model directory not found: {ANIMAGINE_MODEL_DIR}"
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = DiffusionPipeline.from_pretrained(
        str(ANIMAGINE_MODEL_DIR),
        torch_dtype=dtype,
        use_safetensors=True,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 832,
    height: int = 1216,
    steps: int = 28,
    guidance_scale: float = 7.0,
    seed: int = 42,
):
    ensure_animagine_dirs()

    pipe = load_pipeline()
    negative_prompt = negative_prompt or default_negative_prompt()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"animagine_{ts}"

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    prompt_json = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }

    save_json(ANIMAGINE_PROMPTS_DIR / f"{base_name}.json", prompt_json)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    out_path = ANIMAGINE_IMAGES_DIR / f"{base_name}.png"
    image.save(out_path)

    save_json(
        ANIMAGINE_METADATA_DIR / f"{base_name}.json",
        {
            **prompt_json,
            "image_path": str(out_path),
            "model_dir": str(ANIMAGINE_MODEL_DIR),
        },
    )

    return out_path

if __name__ == "__main__":
    prompt = (
        "masterpiece, best quality, monochrome manga, black and white, "
        "gritty psychological suspense, rough ink lines, dramatic shadows, "
        "high contrast, nervous young man in a deep forest, mist, tall trees, "
        "eerie silence, sweat, tense expression, cinematic composition, detailed background"
    )


    out = generate_image(prompt=prompt)
    print(f"saved: {out}")
