#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/team-009/project"
SITE_DIR="${PROJECT_DIR}/site"
RUNTIME_DIR="${SITE_DIR}/runtime"
SLURM_SCRIPT="${SITE_DIR}/run_web.slurm"
export NGROK_AUTHTOKEN="your_key"

mkdir -p "${RUNTIME_DIR}"

echo "======================================"
echo " Submit FastAPI as Slurm batch job"
echo "======================================"

if [[ -f "${RUNTIME_DIR}/web.jobid" ]]; then
  OLD_JOBID="$(cat "${RUNTIME_DIR}/web.jobid" || true)"
  if [[ -n "${OLD_JOBID}" ]]; then
    if squeue -j "${OLD_JOBID}" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "${OLD_JOBID}" -h -o '%A')" ]]; then
      echo "[WARN] existing job is still running: ${OLD_JOBID}"
      echo "[INFO] stop it first if you want restart"
      exit 1
    fi
  fi
fi

SUBMIT_OUTPUT="$(sbatch "${SLURM_SCRIPT}")"
echo "${SUBMIT_OUTPUT}"

JOBID="$(echo "${SUBMIT_OUTPUT}" | awk '{print $4}')"
if [[ -z "${JOBID}" ]]; then
  echo "[ERROR] failed to parse job id"
  exit 1
fi

echo "${JOBID}" > "${RUNTIME_DIR}/web.jobid"
echo "[INFO] submitted job id: ${JOBID}"