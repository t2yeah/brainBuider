#!/usr/bin/env bash
set -e

PROJECT=/home/team-009/project
VENV=$PROJECT/venv
IMAGE="$HOME/nvidia+pytorch+25.11-py3.sqsh"

echo "=== Slurm経由でコンテナ起動 ==="

srun -p gpu --gres=gpu:1 -c 8 --mem=32G \
  --container-image="$IMAGE" \
  --container-mounts="$HOME:$HOME" \
  --pty bash -lc "
set -e

echo '=== venv activate ==='

if [ ! -d \"$VENV\" ]; then
  echo '⚠ venv がありません'
  echo 'setup_env.sh を先に実行してください'
  exit 1
fi

source \"$VENV/bin/activate\"

echo
echo '✅ 作業環境起動完了'
echo 'HOME='$HOME
echo 'VIRTUAL_ENV='$VIRTUAL_ENV
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('cuda_device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY
echo

exec bash
"