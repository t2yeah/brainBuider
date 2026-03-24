#!/bin/bash
#SBATCH --job-name=webapp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/team-009/project/site/logs/web-%j.out
#SBATCH --error=/home/team-009/project/site/logs/web-%j.err

set -euo pipefail

PROJECT="/home/team-009/project"
SITE_DIR="${PROJECT}/site"
IMAGE="$HOME/nvidia+pytorch+25.11-py3.sqsh"

mkdir -p "${SITE_DIR}/logs" "${SITE_DIR}/runtime"

echo "======================================"
echo " Slurm batch job for FastAPI + ngrok"
echo "======================================"
echo "[INFO] JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] HOSTNAME=$(hostname)"
echo "[INFO] IMAGE=${IMAGE}"
echo "[INFO] START=$(date '+%F %T')"

if [[ ! -f "${SITE_DIR}/container_start_web.sh" ]]; then
  echo "[ERROR] script not found: ${SITE_DIR}/container_start_web.sh"
  exit 1
fi

if [[ ! -x "${SITE_DIR}/container_start_web.sh" ]]; then
  echo "[ERROR] script is not executable: ${SITE_DIR}/container_start_web.sh"
  exit 1
fi

# ここでは bash -lc の多重起動をやめて、起動スクリプトを直接実行する
srun --unbuffered \
  --container-image="${IMAGE}" \
  --container-mounts="$HOME:$HOME" \
  --export=ALL,NGROK_AUTHTOKEN \
  "${SITE_DIR}/container_start_web.sh"