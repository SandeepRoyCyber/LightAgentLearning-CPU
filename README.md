# LightAgentLearning-CPU

This repository is a **CPU-first starter kit** for running [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) on a laptop-class machine (target: **8 GB RAM, no GPU**).

## What this repo gives you

- A lightweight environment bootstrap script (`scripts/bootstrap_cpu.sh`)
- A conservative runtime profile tuned for 8 GB machines (`configs/cpu_8gb.env`)
- A Python launcher that applies CPU-safe defaults (`scripts/run_agent_lightning_cpu.py`)

> ⚠️ `agent-lightning` is still an active project. Some internals can change quickly. This repo focuses on robust defaults and guardrails so you can stay within CPU and memory limits.

---

## 1) Clone both repositories

```bash
git clone https://github.com/microsoft/agent-lightning.git
cd agent-lightning

# Optional: keep this CPU profile repo side-by-side
git clone <this-repo-url> ../LightAgentLearning-CPU
```

---

## 2) Create a CPU-only Python environment

From this repository:

```bash
cd ../LightAgentLearning-CPU
bash scripts/bootstrap_cpu.sh
```

This script:

1. Creates `.venv`
2. Installs CPU-only PyTorch wheels
3. Installs a minimal set of runtime libraries frequently used with Agent Lightning workflows

If you need stricter reproducibility, pin versions in your own lock file after the first successful run.

---

## 3) Apply an 8 GB memory profile

```bash
cp configs/cpu_8gb.env ../agent-lightning/.env.cpu
```

Then in the `agent-lightning` repo:

```bash
set -a
source .env.cpu
set +a
```

Key choices in this profile:

- `TOKENIZERS_PARALLELISM=false` to avoid extra memory pressure
- `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` to avoid CPU oversubscription
- Tiny batch sizes and short context defaults
- Quantization flags for `llama.cpp` style local inference

---

## 4) Use a small local model

For 8 GB RAM, prefer models in the **1B–3B** range with 4-bit quantization.

Suggested practical options:

- `Qwen2.5-1.5B-Instruct` (GGUF Q4)
- `Llama-3.2-1B-Instruct` (GGUF Q4)
- `Phi-3.5-mini` (if context/batch stays conservative)

Avoid 7B+ models unless you significantly reduce context and accept poor throughput.

---

## 5) Launch with CPU-safe defaults

```bash
python scripts/run_agent_lightning_cpu.py \
  --project-root ../agent-lightning \
  --train-script train.py \
  --model-backend llama_cpp \
  --model-path /absolute/path/to/model.gguf
```

The launcher sets conservative defaults (batch size, workers, context, gradient accumulation) and passes through custom args.

---

## 6) CPU tuning checklist (important)

1. **Keep sequence length short** (256–512 to start).
2. **Use tiny micro-batches** (1, maybe 2).
3. **Increase grad accumulation** instead of batch.
4. **Cap dataloader workers** (0–1 on laptops).
5. **Disable heavy telemetry/logging integrations** unless needed.
6. **Prefer evaluation subsets** during development.
7. **Checkpoint less frequently** to reduce I/O and memory churn.

---

## 7) Recommended development loop

1. Run a 20–50 step smoke test.
2. Watch RAM with `htop` or Task Manager.
3. Increase one parameter at a time.
4. Record stable settings in a local profile file.

---

## 8) What “completely CPU friendly” means in practice

You can run the framework on an 8 GB laptop if you:

- use small quantized models,
- reduce context and batch aggressively,
- avoid GPU-only dependencies,
- and accept slower training/inference throughput.

This repo gives you a baseline that prioritizes **stability over speed**.


## 9) Smoke test the CPU launcher

Yes — you can run a smoke test locally without the full framework stack.

```bash
bash scripts/smoke_test_cpu.sh
```

This creates a temporary mock `train.py`, launches `run_agent_lightning_cpu.py`, and verifies CPU-oriented arguments are passed correctly.

---

## 10) Run a lightweight benchmark

Yes — you can run a repeatable CPU baseline benchmark:

```bash
python scripts/benchmark_cpu.py --iterations 30 --width 50000
```

This benchmark is synthetic (math workload) so it is model-independent and runs on any laptop. Use it to compare before/after tuning (threads, power mode, background load).

For model-specific benchmarking, run your normal Agent Lightning eval script with:
- fixed model + quantization
- fixed prompt set
- fixed context length
and compare tokens/sec and latency across runs.

