#!/usr/bin/env bash
set -euo pipefail

# ---- Config (edit if needed) ----
IMAGE="pytorch2511"
MOUNT="/home/team-009:/home/team-009"
HOME_DIR="/home/team-009"
VENV_ACTIVATE="/home/team-009/project/venv/bin/activate"
WORKDIR="/home/team-009/project"
# ---------------------------------

if ! command -v enroot >/dev/null 2>&1; then
  echo "[ERROR] enroot not found. Run this script on the HOST (team-009@voltmind-002), not inside the container."
  exit 1
fi

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "[ERROR] venv not found: ${VENV_ACTIVATE}"
  echo "Hint: python3 -m venv /home/team-009/project/venv"
  exit 1
fi

# Pass all arguments as the command to run in the container.
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command...>"
  echo "Example: $0 python code/infer_panns.py --in_dir data/test --out_csv outputs/baseline.csv"
  exit 1
fi

# Important: set HOME to writable persistent path to avoid /root (read-only) issues
# Also set MPLCONFIGDIR to avoid matplotlib trying to write under /root/.config
exec enroot start \
  --mount "${MOUNT}" \
  --env "HOME=${HOME_DIR}" \
  --env "MPLCONFIGDIR=${WORKDIR}/models/mpl_cache" \
  -r "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    mkdir -p '${WORKDIR}/models/mpl_cache'
    source '${VENV_ACTIVATE}'
    cd '${WORKDIR}'
    echo '[run.sh] HOME='\"\$HOME\"
    echo '[run.sh] VIRTUAL_ENV='\"\$VIRTUAL_ENV\"
    exec \"\$@\"
  " bash "$@"
