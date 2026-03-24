#!/usr/bin/env bash
set -e

PROJECT=/home/team-009/project
VENV=$PROJECT/venv

echo "=== Python環境セットアップ ==="

exec enroot start \
  --mount /home/team-009:/home/team-009 \
  --env HOME=/home/team-009 \
  -r pytorch2511 \
  bash -lc "

set -e

echo === venv確認 ===

if [ ! -d \"$VENV\" ]; then
  echo 'venv 作成中...'
  python3 -m venv $VENV
fi

source $VENV/bin/activate

pip install -U pip

if [ -f \"$PROJECT/requirements.txt\" ]; then
  pip install -r $PROJECT/requirements.txt
else
  pip install torch librosa soundfile pandas matplotlib scikit-learn panns-inference
  pip freeze > $PROJECT/requirements.txt
fi

echo
echo '✅ セットアップ完了'
echo

exec bash
"
