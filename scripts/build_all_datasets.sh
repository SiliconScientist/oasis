#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="${1:-experiment.toml}"

CONFIG_INFO="$(
  python - "$BASE_CONFIG" <<'PY'
from pathlib import Path
import sys

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oasis.dataset_overrides import configured_dataset_profile_tag, configured_dataset_tags

config_path = Path(sys.argv[1])
tags = configured_dataset_tags(config_path)
representative_tag = configured_dataset_profile_tag(config_path)
if representative_tag not in tags:
    representative_tag = tags[0]
for tag in tags:
    print(f"tag:{tag}")
print(f"representative:{representative_tag}")
PY
)"

DATASET_TAGS=()
REPRESENTATIVE_TAG=""
while IFS= read -r line; do
  case "$line" in
    tag:*)
      DATASET_TAGS+=("${line#tag:}")
      ;;
    representative:*)
      REPRESENTATIVE_TAG="${line#representative:}"
      ;;
  esac
done <<EOF
$CONFIG_INFO
EOF

if [ "${#DATASET_TAGS[@]}" -eq 0 ]; then
  echo "No configured datasets found in $BASE_CONFIG" >&2
  exit 1
fi

OVERRIDE_FILE="$(mktemp)"
cleanup() {
  rm -f "$OVERRIDE_FILE"
}
trap cleanup EXIT

for tag in "${DATASET_TAGS[@]}"; do
  printf '[dataset_profile]\ntag = "%s"\n' "$tag" >"$OVERRIDE_FILE"
  echo "==> Running configured experiments for $tag"
  python -m oasis "$BASE_CONFIG" "$OVERRIDE_FILE"
done

printf '[dataset_profile]\ntag = "%s"\n' "$REPRESENTATIVE_TAG" >"$OVERRIDE_FILE"
echo "==> Final pass to assemble all-datasets plots for $REPRESENTATIVE_TAG"
python -m oasis "$BASE_CONFIG" "$OVERRIDE_FILE"
