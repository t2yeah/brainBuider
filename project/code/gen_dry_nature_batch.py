import csv
import json
import time
import uuid
from pathlib import Path

import torch
import soundfile as sf
from diffusers import StableAudioPipeline

MODEL_ID = "stabilityai/stable-audio-open-1.0"

OUT_DIR = Path("data/dry")
META_DIR = Path("data/dry_meta")
MANIFEST_PATH = Path("data/dry_manifest.csv")

SECONDS = 6.0
STEPS = 60
SEEDS_PER_PROMPT = 3  # テスト時は 1〜3、本番は 3
NUM_WAVEFORMS_PER_PROMPT = 1

NEGATIVE_PROMPT = (
    "music, singing, melody, instruments, echo, reverb, reverberant, "
    "ambience, room tone, background noise, crowd, distorted, low quality"
)

PROMPT_SETS = {
    "footsteps_forest": [
        "dry footsteps walking on forest ground covered with dry leaves, close-mic, clean transient, isolated sound, no reverb",
        "dry slow footsteps on soil and leaves in a quiet forest, close microphone recording, isolated, no ambience",
        "single step on dry leaves in forest floor, close recording, dry sound, isolated transient",
        "light footsteps walking across forest path with leaves and soil, clean transient, close-mic, dry recording",
        "footstep crunching leaves on forest ground, isolated sound, no ambience, no reverb",
        "slow footsteps on dirt trail in forest, close mic, dry sound",
        "footsteps stepping on pine needles and leaves, clean transient, isolated",
        "single heavy step on forest soil and leaves, dry recording",
        "soft walking steps on forest path leaves, close mic recording",
        "footsteps on woodland ground leaves and soil, isolated clean sound",
    ],
    "branch_snap": [
        "dry small tree branch snapping in forest, close microphone recording, isolated sound, no reverb",
        "twig breaking sharply in a quiet forest, clean transient, dry recording",
        "single branch crack sound, close mic, isolated transient",
        "small stick snapping in hands, close recording, dry sound",
        "thin branch snapping with sharp transient, isolated",
        "twig cracking dry wood sound, close microphone",
        "small branch breaking suddenly, clean transient, dry audio",
        "dry stick snapping with quick transient, isolated sound",
        "branch snapping wood crack sound close recording",
        "short dry twig break sound isolated",
    ],
    "bird_call": [
        "single bird chirp short call, close microphone recording, dry sound, isolated",
        "short bird call chirp, clean tone, close recording, no ambience",
        "bird chirping single note, isolated sound, no reverb",
        "short bird whistle chirp, clean and dry recording",
        "single small bird call, isolated audio, close mic",
        "bird chirp short tone clean transient",
        "isolated bird chirp call dry sound",
        "single bird tweet sound close recording",
        "short bird call note isolated",
        "small bird chirp sound dry clean audio",
    ],
    "water_drop": [
        "single water droplet falling onto surface, close mic recording, dry sound",
        "isolated water drop plop sound, clean transient",
        "single droplet splash small water drop isolated",
        "water droplet hitting surface clean sound dry recording",
        "isolated single water drip sound",
        "small water drop falling sound clean transient",
        "single drip sound close recording no echo",
        "water droplet plink sound isolated",
        "isolated droplet hitting water surface",
        "single drip water sound dry audio",
    ],
    "wind_gust": [
        "short wind gust blowing air sound isolated, close recording, dry sound",
        "brief wind whoosh air movement sound isolated",
        "wind passing quickly sound close microphone",
        "short burst of wind air movement dry recording",
        "wind gust whoosh clean sound isolated",
        "quick wind blow sound isolated",
        "short airflow gust sound close mic",
        "brief wind movement sound dry",
        "short natural wind gust isolated audio",
        "wind whoosh burst clean sound",
    ],
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] loading model on device={device}, dtype={dtype}")
    pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    return pipe, device, dtype


def save_audio_and_meta(
    audio_tensor,
    sample_rate: int,
    content_label: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    seconds: float,
    steps: int,
    device: str,
    dtype: str,
):
    run_id = uuid.uuid4().hex[:12]
    wav_path = OUT_DIR / f"{content_label}_{run_id}.wav"
    meta_path = META_DIR / f"{content_label}_{run_id}.json"

    audio_np = audio_tensor.T.float().cpu().numpy()
    sf.write(wav_path.as_posix(), audio_np, sample_rate)

    meta = {
        "run_id": run_id,
        "content_label": content_label,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "seconds": seconds,
        "num_inference_steps": steps,
        "model_id": MODEL_ID,
        "sample_rate": sample_rate,
        "device": device,
        "torch_dtype": dtype,
        "output_wav": str(wav_path),
        "qa_status": "pending",
        "qa_note": "",
        "notes": "dry素材。空間化（RIR畳み込み）は後段で実施予定。",
        "created_at_unix": int(time.time()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return wav_path, meta_path, meta


def append_manifest(rows):
    file_exists = MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "file",
                "content_label",
                "prompt",
                "negative_prompt",
                "seed",
                "seconds",
                "num_inference_steps",
                "model_id",
                "sample_rate",
                "qa_status",
            ])
        writer.writerows(rows)


def main():
    ensure_dirs()
    pipe, device, dtype = load_pipeline()
    sample_rate = pipe.vae.sampling_rate

    manifest_rows = []
    total_jobs = sum(len(prompts) * SEEDS_PER_PROMPT for prompts in PROMPT_SETS.values())
    done = 0
    started = time.time()

    for content_label, prompts in PROMPT_SETS.items():
        print(f"[INFO] category: {content_label} / prompts={len(prompts)}")
        for prompt_idx, prompt in enumerate(prompts):
            for seed_offset in range(SEEDS_PER_PROMPT):
                seed = prompt_idx * 100 + seed_offset
                done += 1
                print(f"[INFO] generating {done}/{total_jobs} | {content_label} | seed={seed}")

                generator = torch.Generator(device=device).manual_seed(seed)

                try:
                    result = pipe(
                        prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=STEPS,
                        audio_end_in_s=SECONDS,
                        num_waveforms_per_prompt=NUM_WAVEFORMS_PER_PROMPT,
                        generator=generator,
                    )

                    audio = result.audios[0]  # (channels, samples)

                    wav_path, meta_path, meta = save_audio_and_meta(
                        audio_tensor=audio,
                        sample_rate=sample_rate,
                        content_label=content_label,
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        seed=seed,
                        seconds=SECONDS,
                        steps=STEPS,
                        device=device,
                        dtype=str(dtype),
                    )

                    manifest_rows.append([
                        str(wav_path),
                        meta["content_label"],
                        meta["prompt"],
                        meta["negative_prompt"],
                        meta["seed"],
                        meta["seconds"],
                        meta["num_inference_steps"],
                        meta["model_id"],
                        meta["sample_rate"],
                        meta["qa_status"],
                    ])

                    append_manifest([manifest_rows[-1]])
                    print(f"[OK] wav={wav_path.name} meta={meta_path.name}")

                except torch.cuda.OutOfMemoryError:
                    print("[ERROR] CUDA OOM. Reduce SECONDS or STEPS.")
                    raise
                except Exception as e:
                    print(f"[ERROR] failed: label={content_label} seed={seed} err={e}")

    elapsed = round(time.time() - started, 2)
    print(f"[DONE] total_generated={len(manifest_rows)} elapsed_sec={elapsed}")
    print(f"[DONE] manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()