#!/usr/bin/env bash
# Deploy the committed code (git HEAD) to a Hugging Face Docker Space.
#
# One-time setup:
#   .venv/bin/pip install huggingface_hub
#   .venv/bin/hf auth login          # token with "write" access
# Deploy / redeploy:
#   deploy/hf_space.sh [space-name]   # default: csc8830-computer-vision
#
# The Space needs a README with YAML front matter, so the GitHub README is
# replaced by a generated one in the uploaded copy only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
NAME="${1:-csc8830-computer-vision}"
USER_NAME="$("$HF" auth whoami --format quiet 2>/dev/null | head -n1 || true)"
if [ -z "$USER_NAME" ]; then
  echo "Not logged in: run $HF auth login" >&2
  exit 1
fi
REPO="$USER_NAME/$NAME"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE"
rm -rf "$STAGE/tests" "$STAGE"/HW1/*.mat
{
  cat <<'EOF'
---
title: CSc 8830 Computer Vision
emoji: 📷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

EOF
  cat "$ROOT/README.md"
} > "$STAGE/README.md"

"$HF" repos create "$REPO" --type space --sdk docker --public --exist-ok
"$HF" upload "$REPO" "$STAGE" . --type space \
  --delete "*" --commit-message "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"

echo "Space:   https://huggingface.co/spaces/$REPO"
echo "App URL: https://$(echo "$USER_NAME-$NAME" | tr '[:upper:]_.' '[:lower:]--').hf.space"
