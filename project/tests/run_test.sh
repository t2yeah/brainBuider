#!/usr/bin/env bash
set -euo pipefail

# このスクリプトの場所を基準に project ルートへ移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[test] project root: ${PROJECT_ROOT}"

# ===== 設定 =====
AUDIO_ID=$(openssl rand -hex 6)
echo "[test] generated audio_id: ${AUDIO_ID}"
TEST_AUDIO="${PROJECT_ROOT}/tests/sample_audio.wav"
UPLOAD_DIR="${PROJECT_ROOT}/data/uploads"
RESULT_DIR="${PROJECT_ROOT}/data/results/${AUDIO_ID}"

# ===== 事前チェック =====
if [ ! -f "${TEST_AUDIO}" ]; then
  echo "[ERROR] test audio not found: ${TEST_AUDIO}"
  exit 1
fi

mkdir -p "${UPLOAD_DIR}"
mkdir -p "${PROJECT_ROOT}/data/results"

# ===== 仮想環境 =====
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
  echo "[test] activate venv"
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/venv/bin/activate"
else
  echo "[WARN] venv not found: ${PROJECT_ROOT}/venv"
  echo "[WARN] continue with system python3"
fi

PYTHON_BIN="$(command -v python || true)"
if [ -z "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[test] python: ${PYTHON_BIN}"

# ===== テスト音声を uploads に配置 =====
cp "${TEST_AUDIO}" "${UPLOAD_DIR}/${AUDIO_ID}.wav"
echo "[test] copied audio -> ${UPLOAD_DIR}/${AUDIO_ID}.wav"

# ===== 前処理（segments 作成） =====
echo "[test] run preprocess: ${AUDIO_ID}"
"${PYTHON_BIN}" -c "
from app.services.audio_preprocess import run
run('${AUDIO_ID}')
"

# ===== 前処理結果確認 =====
SEGMENT_DIR="${PROJECT_ROOT}/data/segments/${AUDIO_ID}"
if [ ! -d "${SEGMENT_DIR}" ]; then
  echo "[ERROR] segment dir not found: ${SEGMENT_DIR}"
  exit 1
fi

echo "[test] segment dir exists: ${SEGMENT_DIR}"
ls -l "${SEGMENT_DIR}"

# ===== パイプライン実行 =====
echo "[test] run pipeline: ${AUDIO_ID}"
"${PYTHON_BIN}" -c "
from app.services.pipeline import run_pipeline
run_pipeline('${AUDIO_ID}')
"

# ===== 結果確認 =====
echo "[test] checking outputs..."

REQUIRED_FILES=(
  \"${RESULT_DIR}/04_features.json\"
  \"${RESULT_DIR}/05_audio_events.json\"
  \"${RESULT_DIR}/05_space_similarity.json\"
  \"${RESULT_DIR}/06_space_judgement.json\"
  \"${RESULT_DIR}/07_scene_interpretation.json\"
  \"${RESULT_DIR}/08_onomatopoeia.json\"
  \"${RESULT_DIR}/08_manga_prompt.json\"
  \"${RESULT_DIR}/09_manga_image.json\"
  \"${RESULT_DIR}/10_manga_text.json\"
  \"${RESULT_DIR}/pipeline_status.json\"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] missing output: $f"
    exit 1
  fi
done

echo "[test] all required files exist"

# final_result は pipeline.py 上では 11_final_result.json のはずなので両対応
if [ -f "${RESULT_DIR}/11_final_result.json" ]; then
  echo "[test] found final result: ${RESULT_DIR}/11_final_result.json"
elif [ -f "${RESULT_DIR}/10_final_result.json" ]; then
  echo "[test] found final result: ${RESULT_DIR}/10_final_result.json"
else
  echo "[WARN] final result json not found"
fi

echo "[test] success"