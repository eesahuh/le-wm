#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HF_MODEL_REPO="${HF_MODEL_REPO:-elysia111/piperx-lewm-checkpoints}"

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

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  if [[ -d "./outputs" ]]; then
    CHECKPOINT_DIR="./outputs"
  else
    cat >&2 <<'EOF'
No ./outputs directory found.

Set CHECKPOINT_DIR to the directory containing checkpoints/logs, for example:
  CHECKPOINT_DIR=/path/to/checkpoints scripts/hf_upload_checkpoints.sh
EOF
    exit 1
  fi
fi

require_login() {
  if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli whoami >/dev/null 2>&1; then
    return 0
  fi

  cat >&2 <<'EOF'
Not logged in to Hugging Face.

Run one of:
  huggingface-cli login
  hf auth login

Then retry this script.
EOF
  exit 1
}

select_upload_cli() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    help_text="$(huggingface-cli upload --help 2>&1 || true)"
    if printf '%s\n' "${help_text}" | grep -qi "usage: .*huggingface-cli.* upload"; then
      HF_UPLOAD_CMD=(huggingface-cli upload)
      return
    fi
  fi
  if command -v hf >/dev/null 2>&1; then
    HF_UPLOAD_CMD=(hf upload)
    return
  fi
  echo "Could not find huggingface-cli or hf. Install huggingface_hub first." >&2
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"

SOURCE_DIR="$("${PYTHON_BIN}" - "${CHECKPOINT_DIR}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ "${SOURCE_DIR}" == "${REPO_ROOT}" ]]; then
  echo "Refusing to upload the repository root as checkpoints." >&2
  echo "Set CHECKPOINT_DIR to a specific outputs/checkpoints directory." >&2
  exit 1
fi

require_login
HF_UPLOAD_CMD=()
select_upload_cli

echo "Uploading checkpoints/logs:"
echo "  local: ${SOURCE_DIR}"
echo "  repo:  ${HF_MODEL_REPO}"

"${HF_UPLOAD_CMD[@]}" "${HF_MODEL_REPO}" "${SOURCE_DIR}" . \
  --repo-type model \
  --exclude ".cache/*" \
  --exclude ".venv/*" \
  --exclude "lerobot_data/*" \
  --exclude "static_eraser_pnp_v30_fps30/*" \
  --exclude "*.parquet" \
  --exclude "*.mp4" \
  --exclude "*.tar.gz" \
  --commit-message "Upload PiperX LeWM checkpoints"
