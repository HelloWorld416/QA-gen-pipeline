#!/usr/bin/env bash
# CPU decoding + remote API; this job does not need a GPU.
set -Eeuo pipefail
set +x
umask 077
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${VIDEO_DIR:?Set VIDEO_DIR or configure .env}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL or configure .env}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-python3}"
export PYTHONUNBUFFERED=1
exec "$PIPELINE_PYTHON" -m streamdrivebench \
 --video-dir "$VIDEO_DIR" --model "$OPENAI_MODEL" \
 --output "${OUTPUT_DIR:-outputs/streamdrivebench}" --workers "${WORKERS:-4}" "$@"
