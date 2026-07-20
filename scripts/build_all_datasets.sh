#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="${1:-experiment.toml}"

DATASET_TAGS=(
  mamun_oh
  khlohc
  rodrigo
  bio_mass
  comer
  func_group
  game_net
  oc20_dense_nhx
  oc20_dense_ocxhx
)

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

printf '[dataset_profile]\ntag = "%s"\n' "mamun_oh" >"$OVERRIDE_FILE"
echo "==> Final pass to assemble all-datasets plots"
python -m oasis "$BASE_CONFIG" "$OVERRIDE_FILE"
