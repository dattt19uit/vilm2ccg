"""
llm_interface.py – LLM interface with Chain-of-Thought reasoning.

Supports two backends:
  1. OpenAI-compatible API  (set OPENAI_API_KEY / OPENAI_BASE_URL env vars)
  2. Rule-based fallback    (always works, no API key required)

The Chain-of-Thought prompt guides the model through:
  Step 1 – identify circuit type
  Step 2 – list inputs / outputs
  Step 3 – describe internal logic
  Step 4 – produce CircuitJSON
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from vilm2ccg.circuit_json import CircuitJSON, CircuitEdge, CircuitNode
from vilm2ccg.input_module import ParsedIntent, parse_vietnamese_prompt


# ---------------------------------------------------------------------------
# System prompt template for CoT
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là một chuyên gia thiết kế mạch số.
Nhiệm vụ của bạn là chuyển đổi mô tả mạch tổ hợp bằng tiếng Việt sang định dạng CircuitJSON.

Hãy thực hiện theo các bước sau (Chain-of-Thought):
1. Xác định loại mạch và số lượng đầu vào/đầu ra.
2. Liệt kê tất cả các cổng logic cần thiết.
3. Mô tả kết nối giữa các cổng.
4. Xuất ra CircuitJSON hợp lệ.

CircuitJSON format:
{
  "circuit_name": "<tên mạch>",
  "description": "<mô tả>",
  "circuit_type": "<loại>",
  "nodes": [
    {"id": "<id>", "type": "<input|output|and|or|not|nand|nor|xor|xnor|mux|buf>",
     "name": "<tên>", "width": 1, "num_inputs": <n>}
  ],
  "edges": [
    {"from": "<id>", "to": "<id>", "from_port": 0, "to_port": 0, "name": "<net>"}
  ]
}

Chỉ trả về JSON, không có giải thích thêm.
"""


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------


class LLMInterface:
    """Interface to a language model for circuit reasoning."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
        use_fallback: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.use_fallback = use_fallback
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # type: ignore
                self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                self._client = None
        return self._client

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def generate_circuit(self, prompt: str) -> CircuitJSON:
        """Convert a Vietnamese prompt to a CircuitJSON object.

        Attempts LLM API call first; falls back to rule-based generation.
        """
        intent = parse_vietnamese_prompt(prompt)

        if self.api_key and not self.use_fallback:
            try:
                return self._call_llm(prompt, intent)
            except Exception:  # noqa: BLE001
                pass

        return build_circuit_from_intent(intent)

    def _call_llm(self, prompt: str, intent: ParsedIntent) -> CircuitJSON:
        """Call the LLM API and parse the returned JSON."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("openai package not installed")

        user_msg = (
            f"Prompt: {prompt}\n\n"
            f"Thông tin phân tích sơ bộ:\n"
            f"- Loại mạch: {intent.circuit_type}\n"
            f"- Số đầu vào: {intent.num_inputs}\n"
            f"- Độ rộng bit: {intent.data_width}\n\n"
            "Hãy tạo CircuitJSON cho mạch này."
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])

        return CircuitJSON.from_json(content)


# ---------------------------------------------------------------------------
# Rule-based circuit builder  (the "always available" fallback)
# ---------------------------------------------------------------------------


def build_circuit_from_intent(intent: ParsedIntent) -> CircuitJSON:
    """Build a CircuitJSON from a :class:`ParsedIntent` without an LLM."""
    builders = {
        "and": _build_basic_gate,
        "or": _build_basic_gate,
        "not": _build_basic_gate,
        "buf": _build_basic_gate,
        "nand": _build_basic_gate,
        "nor": _build_basic_gate,
        "xor": _build_basic_gate,
        "xnor": _build_basic_gate,
        "mux": _build_mux,
        "half_adder": _build_half_adder,
        "full_adder": _build_full_adder,
        "adder": _build_full_adder,
        "decoder": _build_decoder,
        "encoder": _build_encoder,
        "comparator": _build_comparator,
    }
    builder = builders.get(intent.circuit_type, _build_generic)
    return builder(intent)


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------


def _build_basic_gate(intent: ParsedIntent) -> CircuitJSON:
    ctype = intent.circuit_type
    n = intent.num_inputs
    if ctype in {"not", "buf"}:
        n = 1

    cj = CircuitJSON(
        circuit_name=intent.circuit_name,
        description=intent.description or f"{ctype.upper()} gate with {n} input(s)",
        circuit_type=ctype,
    )

    # inputs
    for i in range(n):
        cj.add_node(CircuitNode(id=f"in_{i}", type="input", name=f"A{i}", width=1))

    # gate
    cj.add_node(CircuitNode(id="gate0", type=ctype, name=f"{ctype.upper()}0", width=1, num_inputs=n))

    # output
    cj.add_node(CircuitNode(id="out_0", type="output", name="Y", width=1))

    # edges: inputs → gate
    for i in range(n):
        cj.add_edge(CircuitEdge(from_node=f"in_{i}", to_node="gate0", to_port=i, name=f"w_in{i}"))

    # gate → output
    cj.add_edge(CircuitEdge(from_node="gate0", to_node="out_0", name="w_y"))

    return cj


def _build_mux(intent: ParsedIntent) -> CircuitJSON:
    n_sel = intent.num_select
    n_data = 2 ** n_sel  # number of data inputs

    cj = CircuitJSON(
        circuit_name=f"mux_{n_data}to1",
        description=f"{n_data}-to-1 Multiplexer",
        circuit_type="mux",
    )

    # data inputs
    for i in range(n_data):
        cj.add_node(CircuitNode(id=f"in_{i}", type="input", name=f"I{i}", width=1))

    # select inputs
    for s in range(n_sel):
        cj.add_node(CircuitNode(id=f"sel_{s}", type="input", name=f"S{s}", width=1))

    # MUX gate (abstract)
    cj.add_node(
        CircuitNode(
            id="mux0",
            type="mux",
            name=f"MUX_{n_data}to1",
            width=1,
            num_inputs=n_data + n_sel,
            metadata={"n_data": n_data, "n_sel": n_sel},
        )
    )

    # output
    cj.add_node(CircuitNode(id="out_0", type="output", name="Y", width=1))

    # data edges
    for i in range(n_data):
        cj.add_edge(CircuitEdge(from_node=f"in_{i}", to_node="mux0", to_port=i, name=f"w_d{i}"))

    # select edges
    for s in range(n_sel):
        cj.add_edge(
            CircuitEdge(from_node=f"sel_{s}", to_node="mux0", to_port=n_data + s, name=f"w_s{s}")
        )

    # output edge
    cj.add_edge(CircuitEdge(from_node="mux0", to_node="out_0", name="w_y"))

    return cj


def _build_half_adder(intent: ParsedIntent) -> CircuitJSON:
    cj = CircuitJSON(
        circuit_name="half_adder",
        description="Half Adder (1-bit)",
        circuit_type="half_adder",
    )
    cj.add_node(CircuitNode(id="in_a", type="input", name="A", width=1))
    cj.add_node(CircuitNode(id="in_b", type="input", name="B", width=1))
    cj.add_node(CircuitNode(id="xor0", type="xor", name="XOR0", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="and0", type="and", name="AND0", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="out_sum", type="output", name="SUM", width=1))
    cj.add_node(CircuitNode(id="out_cout", type="output", name="COUT", width=1))

    # XOR for SUM
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="xor0", to_port=0, name="w_a_xor"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="xor0", to_port=1, name="w_b_xor"))
    cj.add_edge(CircuitEdge(from_node="xor0", to_node="out_sum", name="w_sum"))

    # AND for CARRY
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="and0", to_port=0, name="w_a_and"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="and0", to_port=1, name="w_b_and"))
    cj.add_edge(CircuitEdge(from_node="and0", to_node="out_cout", name="w_cout"))

    return cj


def _build_full_adder(intent: ParsedIntent) -> CircuitJSON:
    cj = CircuitJSON(
        circuit_name="full_adder",
        description="Full Adder (1-bit with carry-in)",
        circuit_type="full_adder",
    )
    for inp, nm in [("in_a", "A"), ("in_b", "B"), ("in_cin", "CIN")]:
        cj.add_node(CircuitNode(id=inp, type="input", name=nm, width=1))

    cj.add_node(CircuitNode(id="xor0", type="xor", name="XOR0", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="xor1", type="xor", name="XOR1", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="and0", type="and", name="AND0", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="and1", type="and", name="AND1", width=1, num_inputs=2))
    cj.add_node(CircuitNode(id="or0", type="or", name="OR0", width=1, num_inputs=2))

    cj.add_node(CircuitNode(id="out_sum", type="output", name="SUM", width=1))
    cj.add_node(CircuitNode(id="out_cout", type="output", name="COUT", width=1))

    # XOR0: A XOR B
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="xor0", to_port=0, name="w_a"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="xor0", to_port=1, name="w_b"))

    # XOR1: (A XOR B) XOR CIN = SUM
    cj.add_edge(CircuitEdge(from_node="xor0", to_node="xor1", to_port=0, name="w_ab"))
    cj.add_edge(CircuitEdge(from_node="in_cin", to_node="xor1", to_port=1, name="w_cin"))
    cj.add_edge(CircuitEdge(from_node="xor1", to_node="out_sum", name="w_sum"))

    # AND0: A AND B
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="and0", to_port=0, name="w_a_and"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="and0", to_port=1, name="w_b_and"))

    # AND1: (A XOR B) AND CIN
    cj.add_edge(CircuitEdge(from_node="xor0", to_node="and1", to_port=0, name="w_ab_and"))
    cj.add_edge(CircuitEdge(from_node="in_cin", to_node="and1", to_port=1, name="w_cin_and"))

    # OR0: COUT = (A AND B) OR ((A XOR B) AND CIN)
    cj.add_edge(CircuitEdge(from_node="and0", to_node="or0", to_port=0, name="w_c1"))
    cj.add_edge(CircuitEdge(from_node="and1", to_node="or0", to_port=1, name="w_c2"))
    cj.add_edge(CircuitEdge(from_node="or0", to_node="out_cout", name="w_cout"))

    return cj


def _build_decoder(intent: ParsedIntent) -> CircuitJSON:
    n_in = intent.num_inputs
    n_out = 2 ** n_in

    cj = CircuitJSON(
        circuit_name=f"decoder_{n_in}to{n_out}",
        description=f"{n_in}-to-{n_out} Decoder",
        circuit_type="decoder",
    )

    for i in range(n_in):
        cj.add_node(CircuitNode(id=f"in_{i}", type="input", name=f"A{i}", width=1))
        cj.add_node(
            CircuitNode(id=f"not_{i}", type="not", name=f"NOT{i}", width=1, num_inputs=1)
        )
        cj.add_edge(CircuitEdge(from_node=f"in_{i}", to_node=f"not_{i}", name=f"w_a{i}"))

    for out_idx in range(n_out):
        gate_id = f"and_{out_idx}"
        cj.add_node(
            CircuitNode(id=gate_id, type="and", name=f"AND{out_idx}", width=1, num_inputs=n_in)
        )
        cj.add_node(
            CircuitNode(id=f"out_{out_idx}", type="output", name=f"Y{out_idx}", width=1)
        )
        for bit in range(n_in):
            if (out_idx >> bit) & 1:
                src = f"in_{bit}"
            else:
                src = f"not_{bit}"
            cj.add_edge(
                CircuitEdge(from_node=src, to_node=gate_id, to_port=bit, name=f"w_{out_idx}_{bit}")
            )
        cj.add_edge(CircuitEdge(from_node=gate_id, to_node=f"out_{out_idx}", name=f"w_y{out_idx}"))

    return cj


def _build_encoder(intent: ParsedIntent) -> CircuitJSON:
    n_in = 2 ** intent.num_inputs
    n_out = intent.num_inputs

    cj = CircuitJSON(
        circuit_name=f"encoder_{n_in}to{n_out}",
        description=f"{n_in}-to-{n_out} Priority Encoder",
        circuit_type="encoder",
    )

    for i in range(n_in):
        cj.add_node(CircuitNode(id=f"in_{i}", type="input", name=f"I{i}", width=1))

    for o in range(n_out):
        lines = [i for i in range(n_in) if (i >> o) & 1]
        gate_id = f"or_{o}"
        cj.add_node(
            CircuitNode(id=gate_id, type="or", name=f"OR{o}", width=1, num_inputs=len(lines))
        )
        for port, src_i in enumerate(lines):
            cj.add_edge(
                CircuitEdge(from_node=f"in_{src_i}", to_node=gate_id, to_port=port,
                            name=f"w_i{src_i}_o{o}")
            )
        cj.add_node(CircuitNode(id=f"out_{o}", type="output", name=f"Y{o}", width=1))
        cj.add_edge(CircuitEdge(from_node=gate_id, to_node=f"out_{o}", name=f"w_y{o}"))

    return cj


def _build_comparator(intent: ParsedIntent) -> CircuitJSON:
    """1-bit magnitude comparator (A==B, A>B, A<B)."""
    cj = CircuitJSON(
        circuit_name="comparator_1bit",
        description="1-bit Magnitude Comparator",
        circuit_type="comparator",
    )
    cj.add_node(CircuitNode(id="in_a", type="input", name="A", width=1))
    cj.add_node(CircuitNode(id="in_b", type="input", name="B", width=1))

    # A == B: XNOR
    cj.add_node(CircuitNode(id="xnor0", type="xnor", name="XNOR0", width=1, num_inputs=2))
    # A > B: A AND NOT(B)
    cj.add_node(CircuitNode(id="not0", type="not", name="NOT0", width=1, num_inputs=1))
    cj.add_node(CircuitNode(id="and0", type="and", name="AND0", width=1, num_inputs=2))
    # A < B: NOT(A) AND B
    cj.add_node(CircuitNode(id="not1", type="not", name="NOT1", width=1, num_inputs=1))
    cj.add_node(CircuitNode(id="and1", type="and", name="AND1", width=1, num_inputs=2))

    cj.add_node(CircuitNode(id="out_eq", type="output", name="EQ", width=1))
    cj.add_node(CircuitNode(id="out_gt", type="output", name="GT", width=1))
    cj.add_node(CircuitNode(id="out_lt", type="output", name="LT", width=1))

    # EQ = A XNOR B
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="xnor0", to_port=0, name="w_a_xnor"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="xnor0", to_port=1, name="w_b_xnor"))
    cj.add_edge(CircuitEdge(from_node="xnor0", to_node="out_eq", name="w_eq"))

    # GT = A AND NOT(B)
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="not0", name="w_b_not"))
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="and0", to_port=0, name="w_a_gt"))
    cj.add_edge(CircuitEdge(from_node="not0", to_node="and0", to_port=1, name="w_nb_gt"))
    cj.add_edge(CircuitEdge(from_node="and0", to_node="out_gt", name="w_gt"))

    # LT = NOT(A) AND B
    cj.add_edge(CircuitEdge(from_node="in_a", to_node="not1", name="w_a_not"))
    cj.add_edge(CircuitEdge(from_node="not1", to_node="and1", to_port=0, name="w_na_lt"))
    cj.add_edge(CircuitEdge(from_node="in_b", to_node="and1", to_port=1, name="w_b_lt"))
    cj.add_edge(CircuitEdge(from_node="and1", to_node="out_lt", name="w_lt"))

    return cj


def _build_generic(intent: ParsedIntent) -> CircuitJSON:
    """Fallback: build a single-gate circuit of unknown type."""
    cj = CircuitJSON(
        circuit_name="unknown_circuit",
        description=intent.description,
        circuit_type="unknown",
    )
    cj.add_node(CircuitNode(id="in_0", type="input", name="A", width=1))
    cj.add_node(CircuitNode(id="out_0", type="output", name="Y", width=1))
    cj.add_edge(CircuitEdge(from_node="in_0", to_node="out_0", name="w_pass"))
    return cj
