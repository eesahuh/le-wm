#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run scripts/vast_setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

export PIPERX_LEROBOT_ROOT="${PIPERX_LEROBOT_ROOT:-./lerobot_data/static_eraser_pnp_v30_fps30}"
export STABLEWM_HOME="${STABLEWM_HOME:-./outputs/stable_worldmodel}"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-./outputs/stable_pretraining}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-./outputs/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-./outputs/matplotlib}"

mkdir -p "${STABLEWM_HOME}" "${SPT_CACHE_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

DATASET_DIR="$(python - "${PIPERX_LEROBOT_ROOT}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

for required in \
  "meta/info.json" \
  "data/chunk-000/file-000.parquet" \
  "videos/observation.image/chunk-000/file-000.mp4"
do
  if [[ ! -f "${DATASET_DIR}/${required}" ]]; then
    echo "Missing required dataset file: ${DATASET_DIR}/${required}" >&2
    echo "Run scripts/hf_download_dataset.sh or set PIPERX_LEROBOT_ROOT." >&2
    exit 1
  fi
done

CMD=(
  python train.py
  data=piperx_lerobot
  trainer.accelerator="${ACCELERATOR}"
  trainer.devices="${DEVICES}"
  trainer.max_epochs="${MAX_EPOCHS}"
  loader.batch_size="${BATCH_SIZE}"
  num_workers="${NUM_WORKERS}"
  loader.persistent_workers="${PERSISTENT_WORKERS}"
)

if [[ "${NUM_WORKERS}" == "0" ]]; then
  CMD+=(loader.prefetch_factor=null)
fi

CMD+=("$@")

echo "Dataset: ${DATASET_DIR}"
echo "StableWM cache: ${STABLEWM_HOME}"
echo "Stable Pretraining cache: ${SPT_CACHE_DIR}"
echo "Running command:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
