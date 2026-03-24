from pathlib import Path
import mimetypes
import re
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routes.upload import router as upload_router

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # /home/team-009/project
HOME_ROOT = PROJECT_ROOT.parent                      # /home/team-009

RESULT_ROOT = PROJECT_ROOT / "data" / "results"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"

SITE_DIR_CANDIDATES = [
    HOME_ROOT / "site",          # /home/team-009/site
    PROJECT_ROOT / "site",       # /home/team-009/project/site
]

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="SoundSpace Generator API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ngrok / ローカル確認用。必要なら後で絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)

# =========================================================
# Validation
# =========================================================
AUDIO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
SAFE_FILE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def validate_audio_id(audio_id: str) -> str:
    if not AUDIO_ID_PATTERN.fullmatch(audio_id or ""):
        raise HTTPException(status_code=400, detail="invalid audio_id")
    return audio_id


def validate_filename(filename: str) -> str:
    if not SAFE_FILE_PATTERN.fullmatch(filename or ""):
        raise HTTPException(status_code=400, detail="invalid filename")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    return filename


def safe_resolve(base_dir: Path, target: Path) -> Path:
    base = base_dir.resolve()
    resolved = target.resolve()

    if not str(resolved).startswith(str(base)):
        raise HTTPException(status_code=403, detail="forbidden")

    return resolved


def find_site_dir() -> Path | None:
    for d in SITE_DIR_CANDIDATES:
        if d.exists() and d.is_dir():
            return d
    return None


def guess_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


# =========================================================
# Static site
# =========================================================
SITE_DIR = find_site_dir()

if SITE_DIR:
    assets_dir = SITE_DIR / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "result_root": str(RESULT_ROOT),
        "upload_root": str(UPLOAD_ROOT),
        "site_dir": str(SITE_DIR) if SITE_DIR else None,
    }


@app.get("/")
def serve_index():
    site_dir = find_site_dir()
    if not site_dir:
        raise HTTPException(status_code=404, detail="site directory not found")

    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")

    return FileResponse(
        path=str(index_path),
        media_type="text/html; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get("/favicon.ico")
def favicon():
    site_dir = find_site_dir()
    if not site_dir:
        raise HTTPException(status_code=404, detail="favicon not found")

    favicon_path = site_dir / "favicon.ico"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="favicon not found")

    return FileResponse(
        path=str(favicon_path),
        media_type="image/x-icon",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# =========================================================
# Media routes
# =========================================================
@app.api_route("/media/result/{audio_id}/{filename}", methods=["GET", "HEAD"])
def get_result_media(audio_id: str, filename: str):
    audio_id = validate_audio_id(audio_id)
    filename = validate_filename(filename)

    base_dir = RESULT_ROOT / audio_id
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="result directory not found")

    target = safe_resolve(base_dir, base_dir / filename)

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(
        path=str(target),
        media_type=guess_media_type(target),
        filename=target.name,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.api_route("/media/upload/{audio_id}", methods=["GET", "HEAD"])
def get_uploaded_audio(audio_id: str):
    audio_id = validate_audio_id(audio_id)

    candidates = [
        UPLOAD_ROOT / f"{audio_id}.wav",
        UPLOAD_ROOT / f"{audio_id}.mp3",
        UPLOAD_ROOT / f"{audio_id}.m4a",
        UPLOAD_ROOT / f"{audio_id}.ogg",
        UPLOAD_ROOT / f"{audio_id}.flac",
    ]

    for candidate in candidates:
        target = safe_resolve(UPLOAD_ROOT, candidate)
        if target.exists() and target.is_file():
            return FileResponse(
                path=str(target),
                media_type=guess_media_type(target),
                filename=target.name,
                headers={"X-Content-Type-Options": "nosniff"},
            )

    raise HTTPException(status_code=404, detail="audio not found")


@app.get("/media/result/{audio_id}")
def list_result_files(audio_id: str):
    audio_id = validate_audio_id(audio_id)

    base_dir = RESULT_ROOT / audio_id
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="result directory not found")

    files = []
    for p in sorted(base_dir.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "url": f"/media/result/{audio_id}/{p.name}",
            })

    return JSONResponse({
        "audio_id": audio_id,
        "files": files,
    })
    
@app.get("/analysis/{audio_id}/status")
def analysis_status(audio_id: str):

    base = RESULT_ROOT / audio_id
    path = base / "pipeline_status.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="status not found")

    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/analysis/{audio_id}")
def analysis_result(audio_id: str):

    base = RESULT_ROOT / audio_id
    path = base / "10_final_result.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="result not found")

    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))