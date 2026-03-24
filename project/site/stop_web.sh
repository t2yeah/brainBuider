#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/team-009/project"
SITE_DIR="${PROJECT_DIR}/site"
RUNTIME_DIR="${SITE_DIR}/runtime"

JOBID=""
if [[ -f "${RUNTIME_DIR}/web.jobid" ]]; then
  JOBID="$(cat "${RUNTIME_DIR}/web.jobid" || true)"
fi

if [[ -n "${JOBID}" ]]; then
  echo "[INFO] stopping Slurm job ${JOBID}"
  scancel "${JOBID}" || true

  echo "[INFO] waiting job stop..."
  for i in $(seq 1 10); do
    if ! squeue -j "${JOBID}" -h | grep -q .; then
      break
    fi
    sleep 1
  done
fi

echo "[INFO] cleaning local remaining processes..."
if [[ -f "${RUNTIME_DIR}/uvicorn.pid" ]]; then
  PID="$(cat "${RUNTIME_DIR}/uvicorn.pid" || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" || true
    sleep 2
    kill -9 "${PID}" 2>/dev/null || true
  fi
  rm -f "${RUNTIME_DIR}/uvicorn.pid"
fi

if [[ -f "${RUNTIME_DIR}/ngrok.pid" ]]; then
  PID="$(cat "${RUNTIME_DIR}/ngrok.pid" || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" || true
    sleep 2
    kill -9 "${PID}" 2>/dev/null || true
  fi
  rm -f "${RUNTIME_DIR}/ngrok.pid"
fi

pkill -f "uvicorn app.main:app" || true
pkill -f "${SITE_DIR}/ngrok http 8010" || true
fuser -k 8010/tcp 2>/dev/null || true
fuser -k 4040/tcp 2>/dev/null || true

rm -f "${RUNTIME_DIR}/web.jobid"
rm -f "${RUNTIME_DIR}/ngrok_public_url.txt"

echo "[INFO] stop completed"

echo "[INFO] port check"
lsof -i :8010 || true
lsof -i :4040 || true