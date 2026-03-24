#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/team-009/project"
SITE_DIR="${PROJECT_DIR}/site"
RUNTIME_DIR="${SITE_DIR}/runtime"

if [[ ! -f "${RUNTIME_DIR}/web.jobid" ]]; then
  echo "[INFO] no active job id file"
  exit 0
fi

JOBID="$(cat "${RUNTIME_DIR}/web.jobid" || true)"
if [[ -z "${JOBID}" ]]; then
  echo "[INFO] empty job id"
  exit 0
fi

echo "=== squeue ==="
squeue -j "${JOBID}" || true

echo
echo "=== sacct ==="
sacct -j "${JOBID}" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList