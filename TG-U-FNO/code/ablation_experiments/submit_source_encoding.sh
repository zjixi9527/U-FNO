#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

sbatch \
    --output="$SCRIPT_DIR/logs/%x_%A_%a.out" \
    --error="$SCRIPT_DIR/logs/%x_%A_%a.err" \
    "$SCRIPT_DIR/run_source_encoding.sh"
