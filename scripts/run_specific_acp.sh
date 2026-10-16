#!/usr/bin/env bash
# Dataset quotas; CPU decoding and remote API. No GPU needed.
set -Eeuo pipefail
set +x
umask 077
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${DATASET:?Set DATASET: bdd100k, opendv-youtube, or boreas}"
: "${VIDEO_DIR:?Set VIDEO_DIR to the selected dataset video folder}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-python3}"
export PYTHONUNBUFFERED=1
exec "$PIPELINE_PYTHON" -m streamdrivebench.specific \
  --dataset "$DATASET" --video-dir "$VIDEO_DIR" --model "$OPENAI_MODEL" \
  --output "${OUTPUT_DIR:-outputs/specific/$DATASET}" --workers "${WORKERS:-4}" "$@"
