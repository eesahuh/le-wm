#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HF_DATASET_REPO="${HF_DATASET_REPO:-elysia111/piperx-lerobot-static-eraser-pnp}"
PIPERX_LEROBOT_ROOT="${PIPERX_LEROBOT_ROOT:-./lerobot_data/static_eraser_pnp_v30_fps30}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

find_python() {
  if command -v python >/dev/null 2>&1; then
    command -v python
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    echo "Could not find python or python3 on PATH." >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"

"${PYTHON_BIN}" - "${HF_DATASET_REPO}" "${PIPERX_LEROBOT_ROOT}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2]).expanduser().resolve()
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading dataset repo {repo_id} to {local_dir}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
)

required = [
    "meta/info.json",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
]
missing = [path for path in required if not (local_dir / path).is_file()]
if missing:
    print("Downloaded snapshot is missing required files:", file=sys.stderr)
    for path in missing:
        print(f"  - {local_dir / path}", file=sys.stderr)
    raise SystemExit(1)

print("Dataset ready:")
for path in required:
    print(f"  - {local_dir / path}")
PY
