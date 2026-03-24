import json
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_DIR
from app.schemas.upload_schema import UploadResponse
from app.services.audio_preprocess import run as run_audio_preprocess
from app.services.pipeline import run_pipeline


router = APIRouter(prefix="/api", tags=["upload"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_preprocess_status(audio_id: str, status: dict) -> None:
    result_dir = RESULT_ROOT / audio_id
    result_dir.mkdir(parents=True, exist_ok=True)

    status_path = result_dir / "pipeline_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def build_initial_status(audio_id: str) -> dict:
    return {
        "audio_id": audio_id,
        "pipeline_version": "v1",
        "status": "queued",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "finished_at": None,
        "current_step": "queued",
        "steps": [],
        "error": None,
    }


def build_failed_status(audio_id: str, step_name: str, exc: Exception) -> dict:
    return {
        "audio_id": audio_id,
        "pipeline_version": "v1",
        "status": "failed",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "finished_at": now_iso(),
        "current_step": step_name,
        "steps": [
            {
                "step_no": 2,
                "step_name": step_name,
                "state": "failed",
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "output_file": None,
                "message": str(exc),
            }
        ],
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def process_audio_pipeline(audio_id: str) -> None:
    """
    バックグラウンドで実行する処理
    1. preprocess（segments作成）
    2. 04〜10 pipeline
    """
    try:
        # queue状態を先に書いておく
        write_preprocess_status(audio_id, build_initial_status(audio_id))

        # 分割済みsegmentsを生成
        run_audio_preprocess(audio_id)

        # 04〜10 実行
        run_pipeline(audio_id)

    except Exception as exc:
        # preprocess段階やpipeline起動前の失敗も status に残す
        write_preprocess_status(
            audio_id,
            build_failed_status(audio_id, "audio_preprocess_or_pipeline", exc)
        )
        print(f"[upload] background pipeline failed: {audio_id}")
        print(traceback.format_exc())


@router.post("/upload-audio", response_model=UploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイル名がありません。")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"未対応の拡張子です。対応形式: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # 一旦すべて読み込んでサイズ確認
    data = await file.read()
    size_bytes = len(data)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="空ファイルです。")

    if size_bytes > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"ファイルサイズが大きすぎます。最大 {MAX_FILE_SIZE // (1024 * 1024)}MB です。"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    audio_id = uuid.uuid4().hex[:12]
    save_name = f"{audio_id}{ext}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        f.write(data)

    # まず queued 状態を書いておく
    write_preprocess_status(audio_id, build_initial_status(audio_id))

    # バックグラウンドで preprocess → pipeline 実行
    background_tasks.add_task(process_audio_pipeline, audio_id)

    return UploadResponse(
        audio_id=audio_id,
        original_filename=file.filename,
        saved_path=str(save_path.relative_to(PROJECT_ROOT)),
        content_type=file.content_type,
        size_bytes=size_bytes,
        status="pipeline_started",
    )