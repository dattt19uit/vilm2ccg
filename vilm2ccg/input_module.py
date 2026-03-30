"""
input_module.py – Vietnamese text prompt parser.

Converts a Vietnamese natural-language description of a combinational circuit
into a structured ParsedIntent that downstream modules can consume.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vietnamese number words
# ---------------------------------------------------------------------------

_VN_NUMBERS: dict[str, int] = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
    "mười sáu": 16,
    "ba mươi hai": 32,
}


# ---------------------------------------------------------------------------
# Circuit keywords (Vietnamese + English aliases used in practice)
# ---------------------------------------------------------------------------

_CIRCUIT_PATTERNS: list[tuple[str, str]] = [
    # MUX  ──────────────────────────────────────────────────────────────────
    (r"\bmux\b|\bghép kênh\b|\bbộ dồn kênh\b|\bmạch chọn\b", "mux"),
    # Adders  ────────────────────────────────────────────────────────────────
    (r"\bbộ cộng đầy đủ\b|\bfull.?adder\b|\bcộng đầy đủ\b", "full_adder"),
    (r"\bbộ cộng nửa\b|\bhalf.?adder\b|\bcộng nửa\b", "half_adder"),
    (r"\bbộ cộng\b|\badder\b", "adder"),
    # Decoder / Encoder  ─────────────────────────────────────────────────────
    (r"\bgiải mã\b|\bdecoder\b|\bdemux\b|\bbộ giải mã\b", "decoder"),
    (r"\bmã hóa\b|\bencoder\b|\bbộ mã hóa\b", "encoder"),
    # Comparator  ────────────────────────────────────────────────────────────
    (r"\bso sánh\b|\bcomparator\b|\bbộ so sánh\b", "comparator"),
    # Basic gates  ───────────────────────────────────────────────────────────
    (r"\bnxor\b|\bxnor\b|\bxor đảo\b|\bđồng nhất\b", "xnor"),
    (r"\bxor\b|\bkộng modulo\b|\bmod 2\b|\bmod2\b", "xor"),
    (r"\bnand\b|\band đảo\b|\bnot.?and\b", "nand"),
    (r"\bnor\b|\bor đảo\b|\bnot.?or\b", "nor"),
    (r"\bnot\b|\bbộ đảo\b|\bđảo\b|\binverter\b", "not"),
    (r"\band\b|\bvà\b", "and"),
    (r"\bor\b|\bhoặc\b", "or"),
    (r"\bbuf\b|\bbuffer\b|\bđệm\b", "buf"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), ct) for p, ct in _CIRCUIT_PATTERNS]


# ---------------------------------------------------------------------------
# ParsedIntent dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedIntent:
    """Result of parsing a Vietnamese prompt."""

    circuit_type: str = "unknown"
    num_inputs: int = 2
    num_select: int = 1       # for MUX: number of select lines
    data_width: int = 1       # bit-width of data paths
    description: str = ""
    raw_prompt: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def circuit_name(self) -> str:
        parts = [self.circuit_type]
        if self.circuit_type == "mux":
            n = 2 ** self.num_select
            parts = [f"mux_{n}to1"]
        elif self.circuit_type in {"and", "or", "nand", "nor", "xor", "xnor"}:
            parts = [f"{self.circuit_type}_{self.num_inputs}in"]
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_vietnamese_prompt(prompt: str) -> ParsedIntent:
    """Parse *prompt* and return a :class:`ParsedIntent`.

    The parser uses rule-based keyword matching.  It is intentionally kept
    simple so it works without any external API calls or ML models.
    """
    text = prompt.strip()
    intent = ParsedIntent(raw_prompt=text, description=text)

    # 1. Identify circuit type
    for pattern, ctype in _COMPILED_PATTERNS:
        if pattern.search(text):
            intent.circuit_type = ctype
            break

    # 2. Extract numeric parameters ─────────────────────────────────────────
    #    a) arabic digits (e.g. "4-1", "3 đầu vào", "8 bit")
    nums = [int(m) for m in re.findall(r"\b(\d+)\b", text)]

    #    b) Vietnamese number words
    for word, val in sorted(_VN_NUMBERS.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
            nums.append(val)

    # 3. Assign numbers to parameters depending on circuit type ────────────
    if intent.circuit_type == "mux":
        # patterns like "4-1", "2-1", "8-1"
        mux_m = re.search(r"(\d+)\s*[-:×x]\s*1", text)
        if mux_m:
            n = int(mux_m.group(1))
            intent.num_select = max(1, int(math.log2(n)) if n > 1 else 1)
            intent.num_inputs = n
        elif nums:
            n = nums[0]
            intent.num_select = max(1, int(math.log2(n)) if n > 1 else 1)
            intent.num_inputs = n

    elif intent.circuit_type in {"and", "or", "nand", "nor", "xor", "xnor", "not", "buf"}:
        if intent.circuit_type in {"not", "buf"}:
            intent.num_inputs = 1
        else:
            # look for "N đầu vào" or "N-input"
            inp_m = re.search(r"(\d+)\s*(?:đầu vào|input|in)", text, re.IGNORECASE)
            if inp_m:
                intent.num_inputs = int(inp_m.group(1))
            elif nums:
                # biggest number in the prompt
                intent.num_inputs = max(nums)
            else:
                intent.num_inputs = 2

    elif intent.circuit_type in {"decoder", "encoder"}:
        if nums:
            intent.num_inputs = nums[0]
        else:
            intent.num_inputs = 2

    elif intent.circuit_type in {"half_adder", "full_adder", "adder"}:
        intent.num_inputs = 2 if "half" in intent.circuit_type else 3

    # 4. Data width (bit-width)
    width_m = re.search(r"(\d+)\s*bit", text, re.IGNORECASE)
    if width_m:
        intent.data_width = int(width_m.group(1))

    return intent
