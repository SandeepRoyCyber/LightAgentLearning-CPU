#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

DATA_PATH="$TMP_DIR/distill.jsonl"
OUT_DIR="$TMP_DIR/out"

cat > "$DATA_PATH" <<'JSONL'
{"prompt":"What is CPU-first training?","response":"CPU-first training uses small batches and conservative sequence lengths."}
{"prompt":"How to reduce memory pressure?","response":"Use quantized models and low thread counts."}
{"prompt":"Should I use GPU?","response":"For this profile, no. Keep device on cpu and use tiny models."}
JSONL

python "$ROOT_DIR/scripts/train_cpu_distilled_lm.py" train \
  --data "$DATA_PATH" \
  --out-dir "$OUT_DIR" \
  --steps 15 \
  --batch-size 2 \
  --seq-len 96 \
  --d-model 96 \
  --n-layers 2 \
  --n-heads 4 \
  --log-every 5

python "$ROOT_DIR/scripts/train_cpu_distilled_lm.py" generate \
  --checkpoint "$OUT_DIR/cpu_student_model.pt" \
  --tokenizer "$OUT_DIR/tokenizer.json" \
  --prompt "### Prompt\nHow to run on cpu?\n\n### Response\n" \
  --max-new-tokens 30 \
  --temperature 0.7

echo "CPU_STUDENT_SMOKE_OK"
