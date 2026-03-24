from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_PATH = DATA_DIR / "phase2_batch_run_report.json"

sys.path.append(str(BASE_DIR))

from app.services.interpret_service import run_phase2_pipeline  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    results = []
    success = 0
    failure = 0

    for path in sorted(ANALYSIS_DIR.glob("*.json")):
        try:
            data = load_json(path)
            audio_id = data.get("audio_id") or path.stem
            result = run_phase2_pipeline(audio_id=str(audio_id))
            results.append({
                "audio_id": audio_id,
                "status": "success",
                "detail": result,
            })
            success += 1
        except Exception as e:
            results.append({
                "audio_id": path.stem,
                "status": "error",
                "error": str(e),
                "input_path": str(path),
            })
            failure += 1

    report = {
        "analysis_file_count": len(list(ANALYSIS_DIR.glob("*.json"))),
        "success_count": success,
        "failure_count": failure,
        "results": results,
    }

    write_json(OUTPUT_PATH, report)

    print(json.dumps({
        "output_path": str(OUTPUT_PATH),
        "success_count": success,
        "failure_count": failure,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
