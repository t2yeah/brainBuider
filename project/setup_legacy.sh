#!/usr/bin/env bash
set -e

PROJECT=/home/team-009/project
VENV=$PROJECT/venv

echo "=== コンテナ起動 ==="

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
echo '✅ 環境準備完了'
echo 'HOME='$HOME
echo 'VIRTUAL_ENV='$VIRTUAL_ENV
echo

exec bash
"
