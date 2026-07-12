#!/bin/zsh
set -uo pipefail

ROOT="${0:A:h}"
PYTHON="$ROOT/.venv/bin/python"
LOG="$ROOT/.subtitle-tools/logs/runner.log"
mkdir -p "$ROOT/.subtitle-tools/logs" "$ROOT/.subtitle-tools/state"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/.subtitle-tools/hf-cache"
export XDG_CACHE_HOME="$ROOT/.subtitle-tools/cache"

cd "$ROOT"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Offline subtitle run started"
  exit_code=0
  "$PYTHON" -m subtitle_pipeline process || exit_code=1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Offline subtitle run finished with status $exit_code"
  exit "$exit_code"
} 2>&1 | tee -a "$LOG"
