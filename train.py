#!/usr/bin/env python3
"""
train.py – Fine-tune a seq2seq Transformer on the ViLM2CCG dataset.

Trains a character-level Transformer encoder–decoder model entirely from
scratch – no pre-trained weights, no internet connection required.

The model learns to map a Vietnamese NL circuit description to a compact
CircuitJSON (without the embedded Verilog, which is re-generated at inference
time by the existing rule-based HDL generator).

Architecture (defaults): d_model=256, nhead=8, 2+2 layers, FFN=1024 ≈ 3.8 M params
Training: teacher-forcing cross-entropy, AdamW, linear warmup.

GPU support
-----------
The script automatically detects GPUs and uses them when available.

  --gpus 0        # CPU only (override auto-detection)
  --gpus 1        # single GPU (default when 1 GPU present)
  --gpus 2        # 2-GPU DataParallel (requires ≥ 2 CUDA devices)

Usage
-----
    python train.py                             # 1 epoch (default)
    python train.py --epochs 3 --batch 8
    python train.py --gpus 1                    # single GPU
    python train.py --gpus 2 --batch 16        # 2x GPU DataParallel
    python train.py --output models/my_run

Checkpoint saved to ``models/vilm2ccg-t5/`` (model.pt, vocab.json, config.json).
Run inference with:  python inference.py
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall with:  pip install torch")

from vilm2ccg.seq2seq import CharTokenizer, CircuitSeq2Seq

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent
_DATA_DIR = _ROOT / "dataset" / "data"
_TRAIN_FILE = _DATA_DIR / "train.jsonl"
_VAL_FILE = _DATA_DIR / "val.jsonl"
_DEFAULT_OUTPUT = _ROOT / "models" / "vilm2ccg-t5"

# Prefix added to every source prompt (must match inference.py)
_PREFIX = "convert: "


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_target(output_str: str) -> str:
    """Return a compact CircuitJSON string *without* the embedded Verilog field.

    Keeping the training target shorter improves convergence at 1 epoch.
    Verilog is added back by ``inference.py`` via the HDL generator.
    """
    try:
        obj = json.loads(output_str)
        compact = {
            "id":         obj.get("id", ""),
            "input_text": obj.get("input_text", ""),
            "graph":      obj.get("graph", {}),
            "metadata":   obj.get("metadata", {}),
        }
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    except (json.JSONDecodeError, KeyError):
        return output_str


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    """Load (source, target) string pairs from an Alpaca JSONL file."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = _PREFIX + rec.get("input", "")
            tgt = _build_target(rec["output"])
            pairs.append((src, tgt))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: CharTokenizer,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.pairs = pairs
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        src_str, tgt_str = self.pairs[idx]
        src_ids = self.tok.encode(src_str, add_bos=False, add_eos=True)[: self.max_src_len]
        tgt_ids = self.tok.encode(tgt_str, add_bos=True,  add_eos=True)[: self.max_tgt_len]
        return {"src": src_ids, "tgt": tgt_ids}


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_src = max(len(b["src"]) for b in batch)
    max_tgt = max(len(b["tgt"]) for b in batch)
    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        src_padded[i, : len(b["src"])] = torch.tensor(b["src"])
        tgt_padded[i, : len(b["tgt"])] = torch.tensor(b["tgt"])
    return {"src": src_padded, "tgt": tgt_padded}


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model: CircuitSeq2Seq,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    pad_id: int,
) -> float:
    model.train()
    total_loss = total_tokens = 0.0
    for step, batch in enumerate(loader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in, (src == pad_id), (tgt_in == pad_id))
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        n = int((tgt_out != pad_id).sum())
        total_loss += loss.item() * n
        total_tokens += n
        if (step + 1) % 10 == 0:
            print(f"    step {step + 1:>3}/{len(loader)}  loss={loss.item():.4f}", flush=True)
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _eval_epoch(
    model: CircuitSeq2Seq,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = total_tokens = 0.0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in, (src == pad_id), (tgt_in == pad_id))
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        n = int((tgt_out != pad_id).sum())
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train seq2seq Transformer on ViLM2CCG dataset")
    p.add_argument("--output",     default=str(_DEFAULT_OUTPUT),
                   help="Checkpoint directory (default: models/vilm2ccg-t5)")
    p.add_argument("--epochs",     type=int,   default=1,    help="Training epochs (default: 1)")
    p.add_argument("--batch",      type=int,   default=4,    help="Batch size per device (default: 4)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    p.add_argument("--max-src",    type=int,   default=256,  help="Max source token length")
    p.add_argument("--max-tgt",    type=int,   default=1024, help="Max target token length")
    p.add_argument("--d-model",    type=int,   default=256,  help="Transformer d_model")
    p.add_argument("--nhead",      type=int,   default=8,    help="Attention heads")
    p.add_argument("--enc-layers", type=int,   default=2,    help="Encoder layers")
    p.add_argument("--dec-layers", type=int,   default=2,    help="Decoder layers")
    p.add_argument("--gpus",       type=int,   default=-1,
                   help="Number of GPUs to use: 0=CPU, 1=single GPU, 2=DataParallel 2×GPU. "
                        "Default: auto-detect (use all available GPUs up to 2).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GPU usability probe
# ---------------------------------------------------------------------------


def _count_usable_gpus() -> int:
    """Return the number of CUDA devices that can actually execute tensor ops.

    ``torch.cuda.is_available()`` / ``device_count()`` may return True/> 0
    even for GPUs whose compute capability is below the minimum supported by
    the installed PyTorch build (e.g. sm_60 P100 on a PyTorch ≥ 2.0 wheel
    that requires sm_70+).  A lightweight test operation catches that case.
    """
    n = torch.cuda.device_count()
    if n == 0:
        return 0
    usable = 0
    for i in range(n):
        try:
            t = torch.tensor([1.0], device=f"cuda:{i}")
            _ = t + t          # triggers the actual CUDA kernel
            usable += 1
        except Exception:
            print(
                f"  ⚠️  GPU {i} ({torch.cuda.get_device_name(i)}) is not usable "
                f"with this PyTorch build – falling back to CPU for that device.",
                flush=True,
            )
    return usable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device / GPU selection
    # ------------------------------------------------------------------
    n_cuda = _count_usable_gpus()
    requested_gpus = args.gpus if args.gpus >= 0 else min(n_cuda, 2)
    requested_gpus = min(requested_gpus, n_cuda)  # can't use more than available

    if requested_gpus == 0 or n_cuda == 0:
        device = torch.device("cpu")
        use_data_parallel = False
    elif requested_gpus == 1:
        device = torch.device("cuda:0")
        use_data_parallel = False
    else:  # 2+
        device = torch.device("cuda:0")
        use_data_parallel = True

    # Effective batch size = batch * num_gpus
    effective_batch = args.batch * max(1, requested_gpus)

    gpu_info = (
        f"CPU" if device.type == "cpu"
        else f"GPU×{requested_gpus} (DataParallel)" if use_data_parallel
        else "GPU×1"
    )
    print(f"Device : {gpu_info}")
    print(f"Batch  : {args.batch} per device × {max(1, requested_gpus)} = {effective_batch} total")
    print(f"Output : {output_dir}")

    # 1. Data
    print("\n[1/5] Loading dataset…")
    train_pairs = _load_pairs(_TRAIN_FILE)
    val_pairs   = _load_pairs(_VAL_FILE)
    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # 2. Vocabulary
    print("[2/5] Building vocabulary…")
    tokenizer = CharTokenizer()
    tokenizer.build([s for s, _ in train_pairs] + [t for _, t in train_pairs])
    tokenizer.save(output_dir / "vocab.json")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # 3. DataLoaders (use effective batch for the loader)
    collate_fn = functools.partial(_collate, pad_id=tokenizer.pad_id)
    train_dl = DataLoader(
        Seq2SeqDataset(train_pairs, tokenizer, args.max_src, args.max_tgt),
        batch_size=effective_batch, shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        Seq2SeqDataset(val_pairs, tokenizer, args.max_src, args.max_tgt),
        batch_size=effective_batch, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # 4. Model
    print("[3/5] Building model…")
    base_model = CircuitSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
    ).to(device)

    if use_data_parallel:
        gpu_ids = list(range(requested_gpus))
        model: nn.Module = nn.DataParallel(base_model, device_ids=gpu_ids)
        print(f"  DataParallel across GPU ids: {gpu_ids}")
    else:
        model = base_model

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Save architecture config
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.enc_layers,
        "num_decoder_layers": args.dec_layers,
        "dim_feedforward": 1024,
        "dropout": 0.1,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # 5. Training
    print(f"[4/5] Training for {args.epochs} epoch(s)…")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = _train_epoch(model, train_dl, optimizer, scheduler, criterion, device, tokenizer.pad_id)
        val_loss   = _eval_epoch(model, val_dl, criterion, device, tokenizer.pad_id)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({time.time()-t0:.1f}s)")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Unwrap DataParallel before saving so inference.py loads cleanly
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, output_dir / "model.pt")
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    print(f"\n[5/5] Done. Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint : {output_dir}")
    print("  Run inference with:  python inference.py")


if __name__ == "__main__":
    main()
