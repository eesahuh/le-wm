#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo "Could not find python3 or python on PATH." >&2
    exit 1
  fi
}

has_global_cuda_torch() {
  local python_bin="$1"
  "${python_bin}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Repo: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  VENV_ARGS=()
  if has_global_cuda_torch "${PYTHON_BIN}"; then
    echo "Detected CUDA-enabled PyTorch in the base Python; creating venv with system site packages."
    VENV_ARGS+=(--system-site-packages)
  fi
  "${PYTHON_BIN}" -m venv "${VENV_ARGS[@]}" "${VENV_DIR}"
else
  echo "Using existing ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# Do not pass --upgrade here. On many Vast images CUDA PyTorch is already
# installed; plain pip install will reuse satisfying torch/torchvision installs.
python -m pip install \
  'stable-worldmodel[train]' \
  lightning \
  hydra-core \
  stable-pretraining \
  torchvision \
  imageio \
  imageio-ffmpeg \
  pyarrow \
  pillow \
  huggingface_hub \
  'datasets>=5'

python - <<'PY'
import importlib
import importlib.metadata as md

checks = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("lightning", "lightning"),
    ("hydra", "hydra-core"),
    ("stable_pretraining", "stable-pretraining"),
    ("stable_worldmodel", "stable-worldmodel"),
    ("huggingface_hub", "huggingface_hub"),
]

for module_name, dist_name in checks:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = md.version(dist_name)
        except md.PackageNotFoundError:
            version = "unknown"
    print(f"{module_name}: {version}")

import torch

print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
PY

