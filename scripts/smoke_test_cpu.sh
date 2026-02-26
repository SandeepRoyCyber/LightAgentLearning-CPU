#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROJECT_DIR="$TMP_DIR/agent-lightning-mock"
mkdir -p "$PROJECT_DIR"

cat > "$PROJECT_DIR/train.py" <<'PY'
import argparse

p = argparse.ArgumentParser()
p.add_argument('--device', required=True)
p.add_argument('--model_backend', required=True)
p.add_argument('--model_path', required=True)
p.add_argument('--max_seq_len', type=int, required=True)
p.add_argument('--batch_size', type=int, required=True)
p.add_argument('--micro_batch_size', type=int, required=True)
p.add_argument('--grad_accum_steps', type=int, required=True)
p.add_argument('--run_name', default='smoke')
args = p.parse_args()

assert args.device == 'cpu', f"expected cpu, got {args.device}"
assert args.batch_size >= 1
assert args.micro_batch_size >= 1
assert args.grad_accum_steps >= 1
print('SMOKE_TEST_OK')
PY

MODEL_PATH="$TMP_DIR/fake_model.gguf"
touch "$MODEL_PATH"

python "$ROOT_DIR/scripts/run_agent_lightning_cpu.py" \
  --project-root "$PROJECT_DIR" \
  --train-script train.py \
  --model-backend llama_cpp \
  --model-path "$MODEL_PATH" \
  --max-seq-len 256 \
  --batch-size 1 \
  --micro-batch-size 1 \
  --grad-accum-steps 4 \
  -- --run_name smoke
