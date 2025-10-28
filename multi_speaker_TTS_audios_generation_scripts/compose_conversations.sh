#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="./samples"
OUTPUT_DIR="./composed"
COMPOSER="./compose_conversations.py"

# extract unique convo IDs from filenames
mapfile -t CONVOS < <(
  ls "$INPUT_DIR"/*.mp3 2>/dev/null \
    | xargs -n1 basename \
    | sed -E 's/^[a-z]+_([0-9]+)_[0-9]+\.mp3$/\1/' \
    | sort -u
)

echo "Found ${#CONVOS[@]} conversations to compose."

for CID in "${CONVOS[@]}"; do
  echo "==== Processing conversation $CID ===="
  python "$COMPOSER" \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --conv-id     "$CID"
done

