from pathlib import Path
import json
import traceback
import gc
from datetime import datetime
from typing import List, Dict, Any

from app.services.audio_features import run as run_audio_features
from app.services.audio_analyze import run as run_audio_analyze
from app.services.space_similarity import run_space_similarity
from app.services.space_judgement import run as run_space_judgement
from app.services.agent_scene_interpreter import run as run_scene_interpreter
from app.services.agent_onomatopoeia import run as run_onomatopoeia
from app.services.agent_manga_prompt import run as run_manga_prompt
from app.services.generate_manga_image import run as run_generate_manga_image
from app.services.manga_text import run as run_manga_text
from app.services.final_result import run as run_final_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"

def pipeline_cleanup(step_name: str = "") -> None:
    print(f"[pipeline] cleanup start: {step_name}")

    gc.collect()

    try:
        import torch
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

    print(f"[pipeline] cleanup done: {step_name}")

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_result_dir(audio_id: str) -> Path:
    result_dir = RESULT_ROOT / audio_id
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def status_file_path(audio_id: str) -> Path:
    return ensure_result_dir(audio_id) / "pipeline_status.json"


def write_status(audio_id: str, status: Dict[str, Any]) -> None:
    path = status_file_path(audio_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def init_status(audio_id: str) -> Dict[str, Any]:
    return {
        "audio_id": audio_id,
        "pipeline_version": "v2_with_audio_events",
        "status": "initialized",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "finished_at": None,
        "current_step": None,
        "steps": [],
        "error": None,
    }


def load_status(audio_id: str) -> Dict[str, Any]:
    path = status_file_path(audio_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    status = init_status(audio_id)
    write_status(audio_id, status)
    return status


def save_status(audio_id: str, status: Dict[str, Any]) -> None:
    status["updated_at"] = now_iso()
    write_status(audio_id, status)


def update_status(
    status: Dict[str, Any],
    *,
    overall_status: str = None,
    current_step: str = None,
    finished_at: str = None,
    error: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if overall_status is not None:
        status["status"] = overall_status
    if current_step is not None:
        status["current_step"] = current_step
    if finished_at is not None:
        status["finished_at"] = finished_at
    if error is not None:
        status["error"] = error
    status["updated_at"] = now_iso()
    return status


def append_step(
    status: Dict[str, Any],
    *,
    step_no: int,
    step_name: str,
    state: str,
    started_at: str = None,
    finished_at: str = None,
    output_file: str = None,
    message: str = None,
) -> Dict[str, Any]:
    status["steps"].append({
        "step_no": step_no,
        "step_name": step_name,
        "state": state,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_file": output_file,
        "message": message,
    })
    status["updated_at"] = now_iso()
    return status


def update_last_step(
    status: Dict[str, Any],
    *,
    state: str = None,
    finished_at: str = None,
    output_file: str = None,
    message: str = None,
) -> Dict[str, Any]:
    if not status["steps"]:
        return status

    step = status["steps"][-1]
    if state is not None:
        step["state"] = state
    if finished_at is not None:
        step["finished_at"] = finished_at
    if output_file is not None:
        step["output_file"] = output_file
    if message is not None:
        step["message"] = message
    status["updated_at"] = now_iso()
    return status


def build_steps(audio_id: str) -> List[Dict[str, Any]]:
    result_dir = ensure_result_dir(audio_id)

    return [
        {
            "step_no": 4,
            "step_name": "audio_features",
            "runner": run_audio_features,
            "output_file": str(result_dir / "04_features.json"),
        },
        {
            "step_no": 5,
            "step_name": "audio_analyze",
            "runner": run_audio_analyze,
            "output_file": str(result_dir / "05_audio_events.json"),
        },
        {
            "step_no": 6,
            "step_name": "space_similarity",
            "runner": run_space_similarity,
            "output_file": str(result_dir / "05_space_similarity.json"),
        },
        {
            "step_no": 7,
            "step_name": "space_judgement",
            "runner": run_space_judgement,
            "output_file": str(result_dir / "06_space_judgement.json"),
        },
        {
            "step_no": 8,
            "step_name": "scene_interpreter",
            "runner": run_scene_interpreter,
            "output_file": str(result_dir / "07_scene_interpretation.json"),
        },
        {
            "step_no": 9,
            "step_name": "onomatopoeia",
            "runner": run_onomatopoeia,
            "output_file": str(result_dir / "08_onomatopoeia.json"),
        },
        {
            "step_no": 10,
            "step_name": "manga_prompt",
            "runner": run_manga_prompt,
            "output_file": str(result_dir / "08_manga_prompt.json"),
        },
        {
            "step_no": 11,
            "step_name": "generate_manga_image",
            "runner": run_generate_manga_image,
            "output_file": str(result_dir / "09_manga_image.json"),
        },
        {
            "step_no": 12,
            "step_name": "manga_text",
            "runner": run_manga_text,
            "output_file": str(result_dir / "10_manga_text.png"),
        },
        {
            "step_no": 13,
            "step_name": "final_result",
            "runner": run_final_result,
            "output_file": str(result_dir / "11_final_result.json"),
        },
    ]


def run_pipeline(audio_id: str):
    ensure_result_dir(audio_id)
    status = load_status(audio_id)
    steps = build_steps(audio_id)

    update_status(status, overall_status="running", current_step="starting")
    save_status(audio_id, status)

    for step in steps:
        step_no = step["step_no"]
        name = step["step_name"]
        runner = step["runner"]
        output_file = step["output_file"]

        append_step(
            status,
            step_no=step_no,
            step_name=name,
            state="running",
            started_at=now_iso(),
            output_file=output_file,
        )
        update_status(status, overall_status="running", current_step=name)
        save_status(audio_id, status)

        try:
            print(f"[pipeline] running: {name}")
            runner(audio_id)

            update_last_step(
                status,
                state="done",
                finished_at=now_iso(),
                output_file=output_file,
                message="completed",
            )
            update_status(status, overall_status="running", current_step=name)
            save_status(audio_id, status)

        except Exception as e:
            tb = traceback.format_exc()

            update_last_step(
                status,
                state="failed",
                finished_at=now_iso(),
                output_file=output_file,
                message=str(e),
            )
            update_status(
                status,
                overall_status="failed",
                current_step=name,
                finished_at=now_iso(),
                error={
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": tb,
                },
            )
            save_status(audio_id, status)

            print(f"[pipeline] failed: {audio_id} / step={name}")
            print(tb)
            raise

        finally:
            pipeline_cleanup(name)
            
    update_status(
        status,
        overall_status="completed",
        current_step="completed",
        finished_at=now_iso(),
    )
    save_status(audio_id, status)

    print(f"[pipeline] success -> {RESULT_ROOT / audio_id / '10_final_result.json'}")


if __name__ == "__main__":
    run_pipeline("478db4fea37e")