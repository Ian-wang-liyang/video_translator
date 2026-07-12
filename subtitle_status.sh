#!/bin/zsh
set -u
ROOT="${0:A:h}"
exec "$ROOT/.venv/bin/python" -m subtitle_pipeline --json status
