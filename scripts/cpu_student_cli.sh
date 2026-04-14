#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${CPU_STUDENT_MODEL_DIR:-$ROOT_DIR/artifacts/cpu_student}"
CHECKPOINT="${CPU_STUDENT_CHECKPOINT:-$MODEL_DIR/cpu_student_model.pt}"
TOKENIZER="${CPU_STUDENT_TOKENIZER:-$MODEL_DIR/tokenizer.json}"

CMD="${1:-}"
shift || true

case "$CMD" in
  train)
    python "$ROOT_DIR/scripts/train_cpu_distilled_lm.py" train "$@"
    ;;
  generate)
    python "$ROOT_DIR/scripts/train_cpu_distilled_lm.py" generate \
      --checkpoint "$CHECKPOINT" \
      --tokenizer "$TOKENIZER" \
      "$@"
    ;;
  chat)
    python "$ROOT_DIR/scripts/train_cpu_distilled_lm.py" chat \
      --checkpoint "$CHECKPOINT" \
      --tokenizer "$TOKENIZER" \
      "$@"
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash scripts/cpu_student_cli.sh train --data <jsonl> [train args]
  bash scripts/cpu_student_cli.sh generate --prompt "..." [--max-new-tokens 120]
  bash scripts/cpu_student_cli.sh chat [--system-prompt "..."]

Environment overrides:
  CPU_STUDENT_MODEL_DIR   default: artifacts/cpu_student
  CPU_STUDENT_CHECKPOINT  default: $CPU_STUDENT_MODEL_DIR/cpu_student_model.pt
  CPU_STUDENT_TOKENIZER   default: $CPU_STUDENT_MODEL_DIR/tokenizer.json
USAGE
    exit 1
    ;;
esac
