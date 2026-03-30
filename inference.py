#!/usr/bin/env python3
"""
inference.py – Generate CircuitJSON from Vietnamese NL using the fine-tuned model.

Uses the seq2seq Transformer checkpoint from ``train.py``.  If the model
checkpoint does not exist, or the decoded output is not valid JSON, the
script falls back to the rule-based pipeline (same as the Streamlit app).

The Verilog field in the output is always populated by the deterministic
rule-based HDL generator, regardless of whether the model or fallback is used.

Usage
-----
    # Single prompt
    python inference.py --prompt "Tạo mạch AND 2 đầu vào"

    # Interactive demo (4 built-in examples)
    python inference.py

    # From stdin (one prompt per line)
    echo "Cổng XOR 2 đầu vào" | python inference.py --stdin

    # Show raw model output before JSON post-processing
    python inference.py --prompt "Bộ cộng nửa" --raw

    # Custom checkpoint
    python inference.py --model models/my_run --prompt "MUX 4-1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall with:  pip install torch")

from vilm2ccg.seq2seq import CharTokenizer, CircuitSeq2Seq

_ROOT = Path(__file__).parent
_DEFAULT_MODEL = _ROOT / "models" / "vilm2ccg-t5"
_PREFIX = "convert: "   # must match train.py


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def _decode_greedy(
    model: CircuitSeq2Seq,
    tokenizer: CharTokenizer,
    src_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(src, tgt)
        next_id = int(logits[0, -1].argmax())
        tgt = torch.cat([tgt, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break
    return tgt[0].tolist()


@torch.no_grad()
def _decode_beam(
    model: CircuitSeq2Seq,
    tokenizer: CharTokenizer,
    src_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    beam_width: int = 4,
) -> list[int]:
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    beams: list[tuple[float, list[int]]] = [(0.0, [tokenizer.bos_id])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_new_tokens):
        candidates: list[tuple[float, list[int]]] = []
        for score, tgt_ids in beams:
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            logits = model(src, tgt)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            top_ids = log_probs.topk(beam_width).indices.tolist()
            for tok_id in top_ids:
                candidates.append((score + log_probs[tok_id].item(), tgt_ids + [tok_id]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for s, ids in candidates[: beam_width]:
            if ids[-1] == tokenizer.eos_id:
                completed.append((s, ids))
            else:
                beams.append((s, ids))
        if not beams:
            break

    all_seqs = completed + beams
    if not all_seqs:
        return [tokenizer.bos_id, tokenizer.eos_id]
    return max(all_seqs, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class ViLM2CCGInference:
    """Wraps the fine-tuned checkpoint and generates CircuitJSON from text."""

    def __init__(
        self,
        model_dir: Path | str = _DEFAULT_MODEL,
        max_new_tokens: int = 512,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.model_available = False
        model_dir = Path(model_dir)

        if not model_dir.exists():
            print(
                f"⚠️  Checkpoint not found: '{model_dir}'\n"
                "   Using rule-based fallback. Train first with:  python train.py",
                file=sys.stderr,
            )
            return

        vocab_path  = model_dir / "vocab.json"
        model_path  = model_dir / "model.pt"
        config_path = model_dir / "config.json"

        if not all(p.exists() for p in (vocab_path, model_path, config_path)):
            print("⚠️  Incomplete checkpoint – using rule-based fallback.", file=sys.stderr)
            return

        print(f"Loading checkpoint from {model_dir} …")
        self.tokenizer = CharTokenizer.load(vocab_path)

        with open(config_path) as f:
            cfg = json.load(f)

        self.model = CircuitSeq2Seq(**cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.model_available = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {n_params:,}  |  Device: {self.device}")

    # ------------------------------------------------------------------
    def generate_raw(self, prompt: str) -> str:
        """Decode the model output as a raw string (before JSON parsing)."""
        if not self.model_available:
            return ""
        src_ids = self.tokenizer.encode(_PREFIX + prompt, add_bos=False, add_eos=True)[:256]
        if self.beam_width > 1:
            out_ids = _decode_beam(
                self.model, self.tokenizer, src_ids,
                self.max_new_tokens, self.device, self.beam_width,
            )
        else:
            out_ids = _decode_greedy(
                self.model, self.tokenizer, src_ids,
                self.max_new_tokens, self.device,
            )
        return self.tokenizer.decode(out_ids)

    def generate_circuit_dict(self, prompt: str) -> dict[str, Any]:
        """Return a complete CircuitJSON dict (with verilog)."""
        raw = self.generate_raw(prompt)
        if raw:
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return _fill_verilog(parsed, prompt)
        # Fallback
        if raw:
            print("⚠️  Model output is not valid JSON – using rule-based fallback.", file=sys.stderr)
        return _fallback_circuit(prompt)

    def generate(self, prompt: str) -> str:
        """Return a pretty-printed CircuitJSON string."""
        return json.dumps(self.generate_circuit_dict(prompt), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace = text.find("{")
    if brace != -1:
        try:
            return json.loads(text[brace:])
        except json.JSONDecodeError:
            pass
    return None


def _fill_verilog(circuit_dict: dict[str, Any], prompt: str) -> dict[str, Any]:
    try:
        from vilm2ccg.circuit_json import CircuitJSON
        from vilm2ccg.hdl_generator import generate_verilog
        cj = CircuitJSON.from_dict(circuit_dict)
        circuit_dict["verilog"] = generate_verilog(cj)
    except Exception:
        pass
    return circuit_dict


def _fallback_circuit(prompt: str) -> dict[str, Any]:
    try:
        from vilm2ccg.input_module import parse_vietnamese_prompt
        from vilm2ccg.llm_interface import build_circuit_from_intent
        from vilm2ccg.hdl_generator import generate_verilog
        intent = parse_vietnamese_prompt(prompt)
        cj = build_circuit_from_intent(intent)
        cj.input_text = prompt
        cj.verilog = generate_verilog(cj)
        return cj.to_dict()
    except Exception as exc:
        return {"error": str(exc), "prompt": prompt}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViLM2CCG inference")
    p.add_argument("--model",      default=str(_DEFAULT_MODEL),
                   help="Checkpoint directory (default: models/vilm2ccg-t5)")
    p.add_argument("--prompt",     default=None,  help="Vietnamese circuit description")
    p.add_argument("--stdin",      action="store_true",
                   help="Read prompts from stdin (one per line)")
    p.add_argument("--raw",        action="store_true",
                   help="Print raw model output before post-processing")
    p.add_argument("--beams",      type=int, default=1,
                   help="Beam width (default: 1 = greedy, use 4+ for higher quality; "
                        "higher values are much slower on CPU)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens (default: 512)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.beams < 1:
        print("--beams must be ≥ 1 (use 1 for greedy decode, 4+ for beam search)", file=sys.stderr)
        sys.exit(1)
    engine = ViLM2CCGInference(
        model_dir=args.model,
        max_new_tokens=args.max_tokens,
        beam_width=args.beams,
    )

    demo_prompts = [
        "Tạo mạch AND 2 đầu vào",
        "Cổng XOR 2 đầu vào",
        "Bộ cộng nửa",
    ]

    if args.stdin:
        for line in sys.stdin:
            prompt = line.strip()
            if not prompt:
                continue
            if args.raw and engine.model_available:
                print(engine.generate_raw(prompt))
            else:
                print(engine.generate(prompt))
        return

    prompts = [args.prompt] if args.prompt else demo_prompts
    if not args.prompt:
        print("No --prompt given. Running demo examples…\n")

    for prompt in prompts:
        print("=" * 60)
        print(f"Prompt: {prompt}")
        if args.raw and engine.model_available:
            raw = engine.generate_raw(prompt)
            print(f"Raw model output:\n{raw[:200]}{'…' if len(raw) > 200 else ''}\n")
        result = engine.generate(prompt)
        d = json.loads(result)
        meta  = d.get("metadata", {})
        graph = d.get("graph", {})
        src   = "model" if engine.model_available else "rule-based fallback"
        print(f"  source      : {src}")
        print(f"  id          : {d.get('id', '')}")
        print(f"  circuit_type: {meta.get('circuit_type', '')}")
        print(f"  inputs/outputs/gates: "
              f"{meta.get('num_inputs', '?')} / {meta.get('num_outputs', '?')} / {meta.get('num_gates', '?')}")
        print(f"  nodes       : {len(graph.get('nodes', []))}")
        print(f"  edges       : {len(graph.get('edges', []))}")
        vlen  = len(d.get("verilog", ""))
        print(f"  verilog     : {f'yes ({vlen} chars)' if vlen else 'no'}")
        print()


if __name__ == "__main__":
    main()
