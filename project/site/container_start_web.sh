#!/usr/bin/env bash
set -euo pipefail

PORT=8010
PROJECT_DIR="/home/team-009/project"
SITE_DIR="${PROJECT_DIR}/site"
PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
NGROK_BIN="${SITE_DIR}/ngrok"
HEALTH_URL="http://127.0.0.1:${PORT}/openapi.json"
NGROK_URL_FILE="${SITE_DIR}/runtime/ngrok_public_url.txt"

FASTAPI_LOG="${SITE_DIR}/logs/fastapi.log"
NGROK_LOG="${SITE_DIR}/logs/ngrok.log"
FASTAPI_PID_FILE="${SITE_DIR}/runtime/uvicorn.pid"
NGROK_PID_FILE="${SITE_DIR}/runtime/ngrok.pid"

echo "======================================"
echo " Starting FastAPI + ngrok in container"
echo "======================================"

mkdir -p "${SITE_DIR}/logs" "${SITE_DIR}/runtime"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] SITE_DIR=${SITE_DIR}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] NGROK_BIN=${NGROK_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -x "${NGROK_BIN}" ]]; then
  echo "[ERROR] ngrok not found or not executable: ${NGROK_BIN}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/app/main.py" ]]; then
  echo "[ERROR] app main not found: ${PROJECT_DIR}/app/main.py"
  exit 1
fi

echo "[INFO] Stopping old processes if any..."
if [[ -f "${FASTAPI_PID_FILE}" ]]; then
  OLD_PID="$(cat "${FASTAPI_PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" || true
    sleep 2
    kill -9 "${OLD_PID}" 2>/dev/null || true
  fi
  rm -f "${FASTAPI_PID_FILE}"
fi

if [[ -f "${NGROK_PID_FILE}" ]]; then
  OLD_PID="$(cat "${NGROK_PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" || true
    sleep 2
    kill -9 "${OLD_PID}" 2>/dev/null || true
  fi
  rm -f "${NGROK_PID_FILE}"
fi

pkill -f "uvicorn app.main:app" || true
pkill -f "${NGROK_BIN} http ${PORT}" || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
fuser -k 4040/tcp 2>/dev/null || true

echo "[INFO] Waiting for ports to be released..."
for p in "${PORT}" 4040; do
  for i in $(seq 1 10); do
    if ! lsof -i :"${p}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
done

echo "[INFO] Starting FastAPI..."
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: > "${FASTAPI_LOG}"
nohup "${PYTHON_BIN}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" \
  >> "${FASTAPI_LOG}" 2>&1 &
FASTAPI_PID=$!
echo "${FASTAPI_PID}" > "${FASTAPI_PID_FILE}"

echo "[INFO] FastAPI PID=${FASTAPI_PID}"
echo "[INFO] Waiting for FastAPI health check..."

FASTAPI_OK=0
for i in $(seq 1 60); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    FASTAPI_OK=1
    break
  fi

  if ! kill -0 "${FASTAPI_PID}" 2>/dev/null; then
    echo "[ERROR] FastAPI process died."
    tail -n 100 "${FASTAPI_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${FASTAPI_OK}" -ne 1 ]]; then
  echo "[ERROR] FastAPI failed health check."
  tail -n 100 "${FASTAPI_LOG}" || true
  exit 1
fi

echo "[INFO] FastAPI started."

echo "[INFO] Starting ngrok..."
echo "[INFO] ===== NGROK BLOCK START =====" | tee -a "${FASTAPI_LOG}"
echo "[INFO] NGROK_BIN=${NGROK_BIN}" | tee -a "${FASTAPI_LOG}"
echo "[INFO] PORT=${PORT}" | tee -a "${FASTAPI_LOG}"
echo "[INFO] NGROK_LOG=${NGROK_LOG}" | tee -a "${FASTAPI_LOG}"
echo "[INFO] NGROK_PID_FILE=${NGROK_PID_FILE}" | tee -a "${FASTAPI_LOG}"
echo "[INFO] NGROK_URL_FILE=${NGROK_URL_FILE}" | tee -a "${FASTAPI_LOG}"

: > "${NGROK_LOG}"
rm -f "${NGROK_URL_FILE}"

NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN:-}"
if [[ -n "${NGROK_AUTHTOKEN}" ]]; then
  echo "[INFO] configuring ngrok authtoken..." | tee -a "${FASTAPI_LOG}"
  "${NGROK_BIN}" config add-authtoken "${NGROK_AUTHTOKEN}" >> "${NGROK_LOG}" 2>&1 || true
else
  echo "[WARN] NGROK_AUTHTOKEN is empty" | tee -a "${FASTAPI_LOG}"
fi

echo "[INFO] launching ngrok..." | tee -a "${FASTAPI_LOG}"
nohup "${NGROK_BIN}" http "${PORT}" --log=stdout --log-level=debug >> "${NGROK_LOG}" 2>&1 &
NGROK_PID=$!
echo "${NGROK_PID}" > "${NGROK_PID_FILE}"

echo "[INFO] ngrok PID=${NGROK_PID}" | tee -a "${FASTAPI_LOG}"
sleep 2

if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
  echo "[ERROR] ngrok process died immediately" | tee -a "${FASTAPI_LOG}"
  tail -n 100 "${NGROK_LOG}" || true
  exit 1
fi

echo "[INFO] Waiting for ngrok API..." | tee -a "${FASTAPI_LOG}"

NGROK_OK=0
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:4040/api/tunnels >/dev/null 2>&1; then
    NGROK_OK=1
    break
  fi

  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    echo "[ERROR] ngrok process died during API wait" | tee -a "${FASTAPI_LOG}"
    tail -n 100 "${NGROK_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${NGROK_OK}" -ne 1 ]]; then
  echo "[ERROR] ngrok failed to start" | tee -a "${FASTAPI_LOG}"
  tail -n 100 "${NGROK_LOG}" || true
  exit 1
fi

PUBLIC_URL="$(
  curl -fsS http://127.0.0.1:4040/api/tunnels \
  | python3 -c 'import sys, json; data=json.load(sys.stdin); print(data["tunnels"][0]["public_url"])'
)"

echo "${PUBLIC_URL}" > "${NGROK_URL_FILE}"
echo "[INFO] Public URL: ${PUBLIC_URL}" | tee -a "${FASTAPI_LOG}"
echo "[INFO] ===== NGROK BLOCK END =====" | tee -a "${FASTAPI_LOG}"

wait "${FASTAPI_PID}"
