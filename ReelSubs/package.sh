#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/dist}"
OUTPUT_FILE="$OUTPUT_DIR/ReelSubs-linux.tar.gz"

mkdir -p "$OUTPUT_DIR"
tar -C "$SCRIPT_DIR" -czf "$OUTPUT_FILE" \
    --transform='s,^,ReelSubs/,' \
    app.py \
    icons/reelsubs.png \
    install.sh \
    package.sh \
    requirements.txt \
    README.md \
    LICENSE \
    test_app.py \
    .gitignore

printf 'Created Linux installer bundle: %s\n' "$OUTPUT_FILE"
