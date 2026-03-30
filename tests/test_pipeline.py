"""
Tests for the ViLM2CCG pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json

import pytest

from vilm2ccg.circuit_json import CircuitJSON, CircuitEdge, CircuitNode
from vilm2ccg.input_module import parse_vietnamese_prompt
from vilm2ccg.llm_interface import build_circuit_from_intent
from vilm2ccg.hdl_generator import generate_verilog
from vilm2ccg.verification import evaluate_circuit, compute_truth_table, run_verification


# ---------------------------------------------------------------------------
# input_module tests
# ---------------------------------------------------------------------------


class TestParseVietnamesePrompt:
    def test_and_gate(self):
        intent = parse_vietnamese_prompt("Mạch AND 3 đầu vào")
        assert intent.circuit_type == "and"
        assert intent.num_inputs == 3

    def test_and_gate_english(self):
        intent = parse_vietnamese_prompt("and 2 inputs")
        assert intent.circuit_type == "and"

    def test_mux_4to1(self):
        intent = parse_vietnamese_prompt("Tạo mạch MUX 4-1")
        assert intent.circuit_type == "mux"
        assert intent.num_inputs == 4
        assert intent.num_select == 2

    def test_mux_2to1(self):
        intent = parse_vietnamese_prompt("MUX 2-1")
        assert intent.circuit_type == "mux"
        assert intent.num_select == 1

    def test_not_gate(self):
        intent = parse_vietnamese_prompt("cổng NOT")
        assert intent.circuit_type == "not"
        assert intent.num_inputs == 1

    def test_xor_gate(self):
        intent = parse_vietnamese_prompt("cổng XOR 2 đầu vào")
        assert intent.circuit_type == "xor"

    def test_half_adder(self):
        intent = parse_vietnamese_prompt("Bộ cộng nửa")
        assert intent.circuit_type == "half_adder"

    def test_full_adder(self):
        intent = parse_vietnamese_prompt("Bộ cộng đầy đủ")
        assert intent.circuit_type == "full_adder"

    def test_decoder(self):
        intent = parse_vietnamese_prompt("Bộ giải mã 2 đầu vào")
        assert intent.circuit_type == "decoder"
        assert intent.num_inputs == 2

    def test_nand(self):
        intent = parse_vietnamese_prompt("cổng NAND")
        assert intent.circuit_type == "nand"

    def test_nor(self):
        intent = parse_vietnamese_prompt("cổng NOR")
        assert intent.circuit_type == "nor"

    def test_comparator(self):
        intent = parse_vietnamese_prompt("bộ so sánh 1 bit")
        assert intent.circuit_type == "comparator"

    def test_or_vn(self):
        intent = parse_vietnamese_prompt("Mạch OR 4 đầu vào")
        assert intent.circuit_type == "or"
        assert intent.num_inputs == 4

    def test_unknown(self):
        intent = parse_vietnamese_prompt("mạch lạ không biết")
        assert intent.circuit_type == "unknown"


# ---------------------------------------------------------------------------
# circuit_json tests
# ---------------------------------------------------------------------------


class TestCircuitJSON:
    def test_serialize_roundtrip(self):
        cj = CircuitJSON(circuit_name="test", description="test circuit", circuit_type="and")
        cj.add_node(CircuitNode(id="in_0", type="input", name="A"))
        cj.add_node(CircuitNode(id="out_0", type="output", name="Y"))
        cj.add_edge(CircuitEdge(from_node="in_0", to_node="out_0", name="w"))

        json_str = cj.to_json()
        restored = CircuitJSON.from_json(json_str)

        assert restored.circuit_name == "test"
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1

    def test_serialize_new_schema_keys(self):
        cj = CircuitJSON(circuit_name="and_2in", description="AND gate", circuit_type="and")
        cj.add_node(CircuitNode(id="in_0", type="input", name="A"))
        cj.add_node(CircuitNode(id="out_0", type="output", name="Y"))
        cj.add_edge(CircuitEdge(from_node="in_0", to_node="out_0", name="w"))

        d = cj.to_dict()
        # Top-level keys
        assert "id" in d
        assert "input_text" in d
        assert "graph" in d
        assert "verilog" in d
        assert "metadata" in d
        # Graph sub-keys
        assert "nodes" in d["graph"]
        assert "edges" in d["graph"]
        # Edge keys use source/target
        edge = d["graph"]["edges"][0]
        assert "source" in edge
        assert "target" in edge
        # Metadata stats
        assert d["metadata"]["num_inputs"] == 1
        assert d["metadata"]["num_outputs"] == 1
        assert d["metadata"]["num_gates"] == 0

    def test_get_inputs_outputs(self):
        cj = CircuitJSON()
        cj.add_node(CircuitNode(id="i0", type="input", name="A"))
        cj.add_node(CircuitNode(id="g0", type="and", name="AND0"))
        cj.add_node(CircuitNode(id="o0", type="output", name="Y"))

        assert len(cj.get_inputs()) == 1
        assert len(cj.get_outputs()) == 1
        assert len(cj.get_gates()) == 1


# ---------------------------------------------------------------------------
# build_circuit_from_intent tests
# ---------------------------------------------------------------------------


class TestBuildCircuitFromIntent:
    def _build(self, prompt: str) -> CircuitJSON:
        intent = parse_vietnamese_prompt(prompt)
        return build_circuit_from_intent(intent)

    def test_and_circuit(self):
        cj = self._build("Mạch AND 2 đầu vào")
        assert cj.circuit_type == "and"
        assert len(cj.get_inputs()) == 2
        assert len(cj.get_outputs()) == 1
        assert len(cj.get_gates()) == 1

    def test_not_circuit(self):
        cj = self._build("cổng NOT")
        assert cj.circuit_type == "not"
        assert len(cj.get_inputs()) == 1

    def test_mux_4to1(self):
        cj = self._build("MUX 4-1")
        assert cj.circuit_type == "mux"
        # 4 data + 2 select = 6 inputs
        assert len(cj.get_inputs()) == 6

    def test_half_adder(self):
        cj = self._build("Bộ cộng nửa")
        assert cj.circuit_type == "half_adder"
        assert len(cj.get_outputs()) == 2  # SUM, COUT

    def test_full_adder(self):
        cj = self._build("Bộ cộng đầy đủ")
        assert cj.circuit_type == "full_adder"
        assert len(cj.get_inputs()) == 3  # A, B, CIN

    def test_decoder_2to4(self):
        cj = self._build("Bộ giải mã 2 đầu vào")
        assert cj.circuit_type == "decoder"
        assert len(cj.get_outputs()) == 4  # 2^2

    def test_comparator(self):
        cj = self._build("bộ so sánh")
        assert cj.circuit_type == "comparator"
        assert len(cj.get_outputs()) == 3  # EQ, GT, LT


# ---------------------------------------------------------------------------
# Verilog generation tests
# ---------------------------------------------------------------------------


class TestVerilogGeneration:
    def test_and_gate_verilog(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        verilog = generate_verilog(cj)
        assert "module" in verilog
        assert "endmodule" in verilog
        assert "assign" in verilog

    def test_verilog_contains_module_name(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        verilog = generate_verilog(cj)
        assert cj.circuit_name.replace("-", "_") in verilog or "and" in verilog.lower()

    def test_not_gate_verilog(self):
        intent = parse_vietnamese_prompt("cổng NOT")
        cj = build_circuit_from_intent(intent)
        verilog = generate_verilog(cj)
        assert "~" in verilog

    def test_half_adder_verilog(self):
        intent = parse_vietnamese_prompt("Bộ cộng nửa")
        cj = build_circuit_from_intent(intent)
        verilog = generate_verilog(cj)
        assert "^" in verilog  # XOR for SUM
        assert "&" in verilog  # AND for COUT

    def test_mux_verilog(self):
        intent = parse_vietnamese_prompt("MUX 2-1")
        cj = build_circuit_from_intent(intent)
        verilog = generate_verilog(cj)
        assert "?" in verilog  # ternary operator


# ---------------------------------------------------------------------------
# Simulation / verification tests
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_and_truth_table(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        result = evaluate_circuit(cj, {"A0": 1, "A1": 1})
        assert result["Y"] == 1

    def test_and_zero(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        result = evaluate_circuit(cj, {"A0": 1, "A1": 0})
        assert result["Y"] == 0

    def test_or_truth_table(self):
        intent = parse_vietnamese_prompt("Mạch OR 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        assert evaluate_circuit(cj, {"A0": 0, "A1": 0})["Y"] == 0
        assert evaluate_circuit(cj, {"A0": 1, "A1": 0})["Y"] == 1

    def test_not_truth_table(self):
        intent = parse_vietnamese_prompt("cổng NOT")
        cj = build_circuit_from_intent(intent)
        assert evaluate_circuit(cj, {"A0": 0})["Y"] == 1
        assert evaluate_circuit(cj, {"A0": 1})["Y"] == 0

    def test_xor_truth_table(self):
        intent = parse_vietnamese_prompt("cổng XOR")
        cj = build_circuit_from_intent(intent)
        assert evaluate_circuit(cj, {"A0": 0, "A1": 0})["Y"] == 0
        assert evaluate_circuit(cj, {"A0": 1, "A1": 0})["Y"] == 1
        assert evaluate_circuit(cj, {"A0": 1, "A1": 1})["Y"] == 0

    def test_nand_truth_table(self):
        intent = parse_vietnamese_prompt("cổng NAND")
        cj = build_circuit_from_intent(intent)
        assert evaluate_circuit(cj, {"A0": 1, "A1": 1})["Y"] == 0
        assert evaluate_circuit(cj, {"A0": 1, "A1": 0})["Y"] == 1

    def test_half_adder_truth(self):
        intent = parse_vietnamese_prompt("Bộ cộng nửa")
        cj = build_circuit_from_intent(intent)
        r = evaluate_circuit(cj, {"A": 1, "B": 1})
        assert r["SUM"] == 0
        assert r["COUT"] == 1

    def test_full_truth_table_rows(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        tt = compute_truth_table(cj)
        assert len(tt) == 4  # 2^2

    def test_verification_pass(self):
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        result = run_verification(cj)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Visualizer smoke test
# ---------------------------------------------------------------------------


class TestVisualizer:
    def test_returns_bytes(self):
        from vilm2ccg.visualizer import visualize_circuit
        intent = parse_vietnamese_prompt("Mạch AND 2 đầu vào")
        cj = build_circuit_from_intent(intent)
        img_bytes = visualize_circuit(cj, fmt="png")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # PNG magic bytes
        assert img_bytes[:4] == b"\x89PNG"

    def test_mux_visualization(self):
        from vilm2ccg.visualizer import visualize_circuit
        intent = parse_vietnamese_prompt("MUX 4-1")
        cj = build_circuit_from_intent(intent)
        img_bytes = visualize_circuit(cj, fmt="png")
        assert len(img_bytes) > 100
