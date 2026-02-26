#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found in PATH" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch wheels
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Minimal runtime set for typical agent-lightning style projects.
python -m pip install \
  transformers \
  datasets \
  accelerate \
  sentencepiece \
  protobuf \
  pyyaml \
  psutil \
  llama-cpp-python

echo "Done. Activate with: source $VENV_DIR/bin/activate"
