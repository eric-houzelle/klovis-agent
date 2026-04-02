#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CAST_FILE="$PROJECT_DIR/assets/demo.cast"
GIF_FILE="$PROJECT_DIR/assets/demo.gif"

mkdir -p "$PROJECT_DIR/assets"

echo "=== Step 1: Record terminal session ==="
echo "Run your command, then press Ctrl+D or type 'exit' when done."
echo ""
asciinema rec "$CAST_FILE" -c "python run.py 'What are the three laws of robotics and why do they matter?'"

echo ""
echo "Recording saved to $CAST_FILE"

echo ""
echo "=== Step 2: Convert to GIF ==="

if command -v agg &>/dev/null; then
    agg --font-size 14 --cols 120 --rows 30 "$CAST_FILE" "$GIF_FILE"
    echo "GIF saved to $GIF_FILE"
else
    echo "agg not found. Install it to convert to GIF:"
    echo ""
    echo "  curl -L -o /usr/local/bin/agg https://github.com/asciinema/agg/releases/latest/download/agg-aarch64-apple-darwin"
    echo "  chmod +x /usr/local/bin/agg"
    echo ""
    echo "Then re-run this script, or convert manually:"
    echo "  agg --font-size 14 --cols 120 --rows 30 $CAST_FILE $GIF_FILE"
fi
