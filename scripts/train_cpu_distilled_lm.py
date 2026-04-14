#!/usr/bin/env python3
"""Train a tiny CPU-first distilled language model from teacher outputs.

This is a Gemma-style *inspired* decoder-only Transformer that is intentionally
small and CPU-friendly. It is designed for experimentation and constrained
hardware, not parity with production foundation models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Example:
    text: str


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        itos = ["<pad>", "<bos>", "<eos>"] + chars
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(ch, self.pad_id) for ch in text)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text())
        itos = data["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln_2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, t = input_ids.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds configured seq_len {self.seq_len}")

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # Causal mask for autoregressive decoding.
        mask = torch.triu(torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        return self.lm_head(x)


def read_jsonl_examples(path: Path) -> list[Example]:
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            text = f"### Prompt\n{prompt}\n\n### Response\n{response}"
            out.append(Example(text=text))
    if not out:
        raise ValueError(f"No training rows found in {path}")
    return out


def build_batches(
    tokenized: list[list[int]],
    batch_size: int,
    seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []

    for _ in range(batch_size):
        sample = random.choice(tokenized)
        if len(sample) < 2:
            continue
        if len(sample) >= seq_len + 1:
            start = random.randint(0, len(sample) - (seq_len + 1))
            chunk = sample[start : start + seq_len + 1]
        else:
            chunk = sample + [pad_id] * (seq_len + 1 - len(sample))

        inp = chunk[:-1]
        tgt = chunk[1:]
        inputs.append(inp)
        targets.append(tgt)

    if not inputs:
        raise RuntimeError("Could not build non-empty batch")

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.mkl_threads))

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl_examples(data_path)
    corpus = "\n".join(ex.text for ex in examples)
    tok = CharTokenizer.from_text(corpus)
    tokenized = [tok.encode(ex.text) for ex in examples]

    device = torch.device("cpu")
    model = TinyDecoderLM(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = build_batches(tokenized, batch_size=args.batch_size, seq_len=args.seq_len, pad_id=tok.pad_id)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), y.reshape(-1), ignore_index=tok.pad_id)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"step={step:04d} loss={loss.item():.4f} ppl={ppl:.2f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "vocab_size": tok.vocab_size,
        },
    }
    torch.save(ckpt, out_dir / "cpu_student_model.pt")
    tok.save(out_dir / "tokenizer.json")
    print(f"Saved model to: {out_dir / 'cpu_student_model.pt'}")
    print(f"Saved tokenizer to: {out_dir / 'tokenizer.json'}")
    return 0


def generate(args: argparse.Namespace) -> int:
    model, tok, cfg = _load_model_and_tokenizer(args.checkpoint, args.tokenizer)
    output = _sample_completion(
        model=model,
        tok=tok,
        cfg=cfg,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(output)
    return 0


def _load_model_and_tokenizer(checkpoint: str, tokenizer: str) -> tuple[TinyDecoderLM, CharTokenizer, dict]:
    ckpt_path = Path(checkpoint).resolve()
    tokenizer_path = Path(tokenizer).resolve()

    tok = CharTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    model = TinyDecoderLM(
        vocab_size=cfg["vocab_size"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    ).cpu()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, cfg


def _sample_completion(
    model: TinyDecoderLM,
    tok: CharTokenizer,
    cfg: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    ids = tok.encode(prompt)
    ids = ids[:-1]

    for _ in range(max_new_tokens):
        context = ids[-cfg["seq_len"] :]
        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
        if next_id == tok.eos_id:
            break

    return tok.decode(ids)


def chat(args: argparse.Namespace) -> int:
    model, tok, cfg = _load_model_and_tokenizer(args.checkpoint, args.tokenizer)
    print("CPU student CLI ready. Type '/exit' to quit.")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")

    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break

        prompt = "### Prompt\n"
        if args.system_prompt:
            prompt += f"{args.system_prompt}\n"
        prompt += f"{user}\n\n### Response\n"
        output = _sample_completion(
            model=model,
            tok=tok,
            cfg=cfg,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if "### Response" in output:
            reply = output.split("### Response", 1)[1].strip()
        else:
            reply = output.strip()
        print(f"model> {reply}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/generate with a tiny CPU-first distilled decoder LM")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a tiny decoder-only student model on CPU")
    t.add_argument("--data", required=True, help="Path to JSONL file with prompt/response keys")
    t.add_argument("--out-dir", default="artifacts/cpu_student", help="Output directory")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--seq-len", type=int, default=256)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--n-layers", type=int, default=4)
    t.add_argument("--n-heads", type=int, default=4)
    t.add_argument("--mlp-ratio", type=float, default=4.0)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--omp-threads", type=int, default=2)
    t.add_argument("--mkl-threads", type=int, default=2)

    g = sub.add_parser("generate", help="Generate from a trained checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=120)
    g.add_argument("--temperature", type=float, default=0.8)

    c = sub.add_parser("chat", help="Interactive CLI chat loop (CPU only)")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--max-new-tokens", type=int, default=120)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--system-prompt", default="")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        return train(args)
    if args.cmd == "generate":
        return generate(args)
    if args.cmd == "chat":
        return chat(args)
    parser.error(f"Unsupported command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
