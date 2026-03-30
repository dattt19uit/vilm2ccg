#!/usr/bin/env python3
"""
build_dataset.py – NL → CCG training dataset generator.

Generates a dataset of (Vietnamese natural-language prompt, CircuitJSON)
pairs for fine-tuning a local LLM (e.g. via LM Studio) on the
ViLM2CCG task.

Usage:
    python dataset/build_dataset.py              # writes to dataset/data/
    python dataset/build_dataset.py --out /path/to/dir
    python dataset/build_dataset.py --seed 99 --val-ratio 0.1 --test-ratio 0.1

Output files (JSONL):
    full.jsonl   – complete dataset (all examples)
    train.jsonl  – training split
    val.jsonl    – validation split
    test.jsonl   – test split

Each JSONL line is a JSON object in **Alpaca instruction-tuning** format:
    {
      "instruction": "<system instruction>",
      "input":       "<Vietnamese NL prompt>",
      "output":      "<CircuitJSON as a compact JSON string>"
    }

A parallel ``*_chat.jsonl`` file is also written in **chat / messages** format
compatible with OpenAI fine-tuning and LM Studio:
    {
      "messages": [
        {"role": "system",    "content": "<system instruction>"},
        {"role": "user",      "content": "<Vietnamese NL prompt>"},
        {"role": "assistant", "content": "<CircuitJSON string>"}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so vilm2ccg can be imported
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from vilm2ccg.input_module import parse_vietnamese_prompt          # noqa: E402
from vilm2ccg.llm_interface import build_circuit_from_intent       # noqa: E402
from vilm2ccg.circuit_json import CircuitJSON                      # noqa: E402
from vilm2ccg.hdl_generator import generate_verilog               # noqa: E402


# ---------------------------------------------------------------------------
# System instruction (same CoT prompt used in production)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "Bạn là một chuyên gia thiết kế mạch số. "
    "Nhiệm vụ của bạn là chuyển đổi mô tả mạch tổ hợp bằng tiếng Việt sang định dạng CircuitJSON. "
    "Chỉ trả về CircuitJSON hợp lệ, không có giải thích thêm."
)

# ---------------------------------------------------------------------------
# Prompt template library
# Each entry: (template_string, circuit_type, params_dict)
# Template placeholders: {n} = num_inputs, {sel} = num_select, {out} = num_outputs
# ---------------------------------------------------------------------------

_TEMPLATES: list[tuple[str, str, dict]] = []


def _add(templates: list[str], circuit_type: str, params: dict) -> None:
    for t in templates:
        _TEMPLATES.append((t, circuit_type, params))


# ── AND gate ────────────────────────────────────────────────────────────────
for _n in [2, 3, 4, 8]:
    _add(
        [
            f"Tạo mạch AND {_n} đầu vào",
            f"Thiết kế cổng AND với {_n} đầu vào",
            f"Cổng logic AND {_n} input",
            f"Mạch cổng AND có {_n} ngõ vào",
            f"Vẽ sơ đồ mạch AND {_n} đầu vào",
            f"Xây dựng cổng AND {_n} đầu vào đơn bit",
            f"AND gate {_n} đầu vào",
            f"Mạch AND {_n}-input",
        ],
        "and",
        {"num_inputs": _n},
    )

# ── OR gate ──────────────────────────────────────────────────────────────────
for _n in [2, 3, 4, 8]:
    _add(
        [
            f"Tạo mạch OR {_n} đầu vào",
            f"Thiết kế cổng OR với {_n} đầu vào",
            f"Cổng logic OR {_n} ngõ vào",
            f"Mạch OR {_n} input",
            f"Mạch cổng hoặc {_n} đầu vào",
            f"Vẽ cổng OR {_n} đầu vào",
        ],
        "or",
        {"num_inputs": _n},
    )

# ── NOT gate ─────────────────────────────────────────────────────────────────
_add(
    [
        "Tạo cổng NOT",
        "Thiết kế bộ đảo (inverter)",
        "Mạch đảo tín hiệu 1 bit",
        "Cổng NOT đơn đầu vào",
        "Vẽ cổng NOT 1 đầu vào",
        "Inverter 1-bit",
        "Bộ đảo logic NOT",
        "Cổng logic NOT",
        "NOT gate đơn",
        "Mạch nghịch đảo NOT",
    ],
    "not",
    {"num_inputs": 1},
)

# ── NAND gate ────────────────────────────────────────────────────────────────
for _n in [2, 3, 4]:
    _add(
        [
            f"Tạo cổng NAND {_n} đầu vào",
            f"Mạch AND đảo {_n} input",
            f"Thiết kế cổng NAND {_n} ngõ vào",
            f"NAND gate {_n} đầu vào",
            f"Cổng NAND {_n}-input",
            f"Mạch NOT-AND {_n} đầu vào",
        ],
        "nand",
        {"num_inputs": _n},
    )

# ── NOR gate ─────────────────────────────────────────────────────────────────
for _n in [2, 3, 4]:
    _add(
        [
            f"Tạo cổng NOR {_n} đầu vào",
            f"Mạch OR đảo {_n} input",
            f"Thiết kế cổng NOR {_n} ngõ vào",
            f"NOR gate {_n} đầu vào",
            f"Cổng NOR {_n}-input",
            f"Mạch NOT-OR {_n} đầu vào",
        ],
        "nor",
        {"num_inputs": _n},
    )

# ── XOR gate ─────────────────────────────────────────────────────────────────
for _n in [2, 3, 4]:
    _add(
        [
            f"Tạo cổng XOR {_n} đầu vào",
            f"Mạch XOR {_n} ngõ vào",
            f"Cổng Exclusive-OR {_n} đầu vào",
            f"XOR gate {_n} input",
            f"Mạch cộng modulo-2 {_n} đầu vào",
            f"Cổng XOR {_n}-input",
        ],
        "xor",
        {"num_inputs": _n},
    )

# ── XNOR gate ────────────────────────────────────────────────────────────────
_add(
    [
        "Tạo cổng XNOR 2 đầu vào",
        "Mạch XNOR 2 input",
        "Cổng đồng nhất (XNOR) 2 đầu vào",
        "XOR đảo 2 ngõ vào",
        "Thiết kế cổng XNOR",
        "XNOR gate 2 đầu vào",
        "Cổng NXOR 2 input",
        "Mạch so sánh bằng 1-bit XNOR",
    ],
    "xnor",
    {"num_inputs": 2},
)

# ── Buffer ───────────────────────────────────────────────────────────────────
_add(
    [
        "Tạo bộ đệm (buffer) 1 bit",
        "Mạch buffer đơn",
        "Cổng BUF 1 đầu vào",
        "Thiết kế bộ đệm tín hiệu",
        "Buffer gate 1-bit",
        "Mạch đệm không đảo",
    ],
    "buf",
    {"num_inputs": 1},
)

# ── MUX ──────────────────────────────────────────────────────────────────────
for _sel in [1, 2, 3]:
    _n_data = 2 ** _sel
    _add(
        [
            f"Tạo mạch MUX {_n_data}-1",
            f"Thiết kế bộ dồn kênh {_n_data} vào 1 ra",
            f"Mạch ghép kênh {_n_data} đầu vào",
            f"MUX {_n_data}-to-1",
            f"Bộ MUX {_n_data}:1 với {_sel} bit chọn",
            f"Mạch chọn {_n_data}-1 (MUX)",
            f"Multiplexer {_n_data} vào 1 ra",
            f"MUX {_n_data}×1 {_sel} đường chọn",
            f"Thiết kế mạch MUX {_n_data} đầu vào 1 đầu ra",
            f"Bộ dồn kênh {_n_data}-1 bit",
        ],
        "mux",
        {"num_select": _sel, "num_inputs": _n_data},
    )

# ── Half Adder ───────────────────────────────────────────────────────────────
_add(
    [
        "Tạo bộ cộng nửa (half adder)",
        "Thiết kế mạch half adder 1 bit",
        "Bộ cộng nửa 1-bit",
        "Mạch cộng nửa hai đầu vào",
        "Half adder không có nhớ vào",
        "Mạch cộng 2 bit 1-bit không có carry-in",
        "Thiết kế half adder với ngõ ra SUM và CARRY",
        "Bộ cộng nửa: ngõ ra tổng và nhớ",
        "Half adder mạch tổ hợp",
        "Mạch cộng đơn giản 2 đầu vào half adder",
        "Cộng nửa: A+B → SUM, COUT",
        "Vẽ sơ đồ half adder",
    ],
    "half_adder",
    {"num_inputs": 2},
)

# ── Full Adder ───────────────────────────────────────────────────────────────
_add(
    [
        "Tạo bộ cộng đầy đủ (full adder)",
        "Thiết kế mạch full adder 1 bit",
        "Bộ cộng đầy đủ với carry-in",
        "Full adder 3 đầu vào (A, B, CIN)",
        "Mạch cộng đầy đủ có nhớ vào",
        "Cộng đầy đủ 1-bit: A+B+CIN → SUM, COUT",
        "Thiết kế full adder với 3 ngõ vào",
        "Bộ cộng đầy đủ mạch tổ hợp",
        "Full adder: tổng và nhớ ra",
        "Mạch cộng 3 bit đầu vào full adder",
        "Vẽ sơ đồ mạch cộng đầy đủ",
        "Full adder với XOR, AND, OR",
    ],
    "full_adder",
    {"num_inputs": 3},
)

# ── Decoder ──────────────────────────────────────────────────────────────────
for _n_in in [2, 3]:
    _n_out = 2 ** _n_in
    _add(
        [
            f"Tạo bộ giải mã {_n_in} vào {_n_out} ra",
            f"Thiết kế mạch decoder {_n_in}-to-{_n_out}",
            f"Decoder {_n_in} đầu vào {_n_out} đầu ra",
            f"Mạch giải mã {_n_in}:{_n_out}",
            f"Bộ giải mã {_n_in} bit sang {_n_out} đường",
            f"Decoder {_n_in}×{_n_out}",
            f"Mạch giải mã nhị phân {_n_in} bit",
            f"Giải mã {_n_in} bit → {_n_out} đầu ra",
        ],
        "decoder",
        {"num_inputs": _n_in},
    )

# ── Encoder ──────────────────────────────────────────────────────────────────
for _n_out in [2, 3]:
    _n_in = 2 ** _n_out
    _add(
        [
            f"Tạo bộ mã hóa {_n_in} vào {_n_out} ra",
            f"Thiết kế encoder {_n_in}-to-{_n_out}",
            f"Encoder ưu tiên {_n_in} đầu vào",
            f"Mạch mã hóa {_n_in}:{_n_out}",
            f"Bộ mã hóa {_n_in} đường vào {_n_out} bit ra",
            f"Priority encoder {_n_in} input",
            f"Encoder {_n_in}×{_n_out}",
        ],
        "encoder",
        {"num_inputs": _n_out},
    )

# ── Comparator ───────────────────────────────────────────────────────────────
_add(
    [
        "Tạo bộ so sánh 1 bit",
        "Thiết kế mạch comparator 1-bit",
        "Bộ so sánh độ lớn 1 bit (A==B, A>B, A<B)",
        "Comparator 1-bit magnitude",
        "Mạch so sánh 2 số 1-bit",
        "Bộ so sánh: A bằng B, A lớn hơn B, A nhỏ hơn B",
        "1-bit magnitude comparator",
        "So sánh hai bit A và B",
        "Mạch logic so sánh 1-bit",
        "Thiết kế comparator với 3 đầu ra EQ, GT, LT",
    ],
    "comparator",
    {"num_inputs": 2},
)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _circuit_to_compact_json(circuit: CircuitJSON) -> str:
    """Return a compact (single-line) JSON string for the circuit."""
    return json.dumps(circuit.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _circuit_to_pretty_json(circuit: CircuitJSON) -> str:
    """Return an indented JSON string for the circuit."""
    return json.dumps(circuit.to_dict(), ensure_ascii=False, indent=2)

def generate_examples() -> Iterator[dict]:
    """Yield one dataset example per template entry."""
    for template, circuit_type, params in _TEMPLATES:
        prompt = template.strip()

        # Parse and override params so the rule-based builder uses exact values
        intent = parse_vietnamese_prompt(prompt)
        intent.circuit_type = circuit_type
        if "num_inputs" in params:
            intent.num_inputs = params["num_inputs"]
        if "num_select" in params:
            intent.num_select = params["num_select"]

        try:
            circuit = build_circuit_from_intent(intent)
        except Exception:
            continue

        # Set the public-facing id and input_text to match the new schema
        circuit.id = circuit.circuit_name
        circuit.input_text = prompt
        # Embed Verilog in the JSON
        circuit.verilog = generate_verilog(circuit)

        output_json = _circuit_to_pretty_json(circuit)

        yield {
            "instruction": SYSTEM_INSTRUCTION,
            "input": prompt,
            "output": output_json,
            # metadata (not written to JSONL output, used for splits)
            "_circuit_type": circuit_type,
        }


def to_chat_format(example: dict) -> dict:
    """Convert an Alpaca example to chat/messages format."""
    return {
        "messages": [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def write_jsonl(path: Path, examples: list[dict], fmt: str = "alpaca") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            if fmt == "chat":
                record = to_chat_format(ex)
            else:
                record = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):>4} examples → {path}")


def split_dataset(
    examples: list[dict],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by circuit_type to keep class balance."""
    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        ct = ex.get("_circuit_type", "unknown")
        by_type.setdefault(ct, []).append(ex)

    train, val, test = [], [], []
    for ct_examples in by_type.values():
        shuffled = ct_examples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train: n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ViLM2CCG NL→CCG training dataset")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data"),
        help="Output directory (default: dataset/data/)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print CircuitJSON in output (default: compact JSON)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("Generating examples…")
    all_examples = list(generate_examples())
    print(f"  Total examples: {len(all_examples)}")

    # Print breakdown by circuit type
    breakdown: dict[str, int] = {}
    for ex in all_examples:
        ct = ex.get("_circuit_type", "?")
        breakdown[ct] = breakdown.get(ct, 0) + 1
    print("  Breakdown:")
    for ct, cnt in sorted(breakdown.items()):
        print(f"    {ct:<20} {cnt:>3}")

    print("\nSplitting…")
    train, val, test = split_dataset(
        all_examples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("\nWriting Alpaca-format JSONL…")
    write_jsonl(out_dir / "full.jsonl", all_examples, fmt="alpaca")
    write_jsonl(out_dir / "train.jsonl", train, fmt="alpaca")
    write_jsonl(out_dir / "val.jsonl", val, fmt="alpaca")
    write_jsonl(out_dir / "test.jsonl", test, fmt="alpaca")

    print("\nWriting chat-format JSONL (LM Studio / OpenAI fine-tuning)…")
    write_jsonl(out_dir / "full_chat.jsonl", all_examples, fmt="chat")
    write_jsonl(out_dir / "train_chat.jsonl", train, fmt="chat")
    write_jsonl(out_dir / "val_chat.jsonl", val, fmt="chat")
    write_jsonl(out_dir / "test_chat.jsonl", test, fmt="chat")

    print("\nDone.")


if __name__ == "__main__":
    main()
