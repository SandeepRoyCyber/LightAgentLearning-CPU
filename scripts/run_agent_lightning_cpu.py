#!/usr/bin/env python3
"""CPU-safe launcher wrapper for Agent Lightning style training scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent Lightning with CPU-first defaults")
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to local agent-lightning repository",
    )
    parser.add_argument(
        "--train-script",
        default="train.py",
        help="Training entrypoint script inside project root",
    )
    parser.add_argument(
        "--model-backend",
        default=os.getenv("ALIGHTNING_BACKEND", "llama_cpp"),
        choices=["llama_cpp", "hf_cpu"],
        help="Inference/model backend",
    )
    parser.add_argument("--model-path", required=True, help="Model path (e.g. GGUF file)")
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("ALIGHTNING_MAX_SEQ_LEN", "512")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("ALIGHTNING_BATCH_SIZE", "1")))
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=int(os.getenv("ALIGHTNING_MICRO_BATCH_SIZE", "1")),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=int(os.getenv("ALIGHTNING_GRAD_ACCUM_STEPS", "8")),
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args forwarded to training script")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    train_script = (project_root / args.train_script).resolve()

    if not project_root.exists():
        print(f"Project root not found: {project_root}", file=sys.stderr)
        return 2

    if not train_script.exists():
        print(f"Train script not found: {train_script}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MALLOC_ARENA_MAX", "2")

    cmd = [
        sys.executable,
        str(train_script),
        "--device",
        "cpu",
        "--model_backend",
        args.model_backend,
        "--model_path",
        args.model_path,
        "--max_seq_len",
        str(args.max_seq_len),
        "--batch_size",
        str(args.batch_size),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
    ]

    if args.extra_args:
        extra_args = args.extra_args
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        cmd.extend(extra_args)

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
