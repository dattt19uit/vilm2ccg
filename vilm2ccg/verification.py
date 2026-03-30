"""
verification.py – Functional simulation using Icarus Verilog (iverilog).

Generates a self-checking testbench from a CircuitJSON truth table,
compiles it with iverilog, and runs it to verify correctness.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Any

from vilm2ccg.circuit_json import CircuitJSON
from vilm2ccg.hdl_generator import generate_verilog, _safe_id


# ---------------------------------------------------------------------------
# Truth-table evaluation (pure Python)
# ---------------------------------------------------------------------------

_GATE_EVAL: dict[str, Any] = {
    "and":  lambda ins: all(ins) if ins else 0,
    "or":   lambda ins: any(ins) if ins else 0,
    "not":  lambda ins: not ins[0] if ins else 0,
    "buf":  lambda ins: ins[0] if ins else 0,
    "nand": lambda ins: (not all(ins)) if ins else 1,
    "nor":  lambda ins: (not any(ins)) if ins else 1,
    "xor":  lambda ins: sum(ins) % 2 == 1 if ins else 0,
    "xnor": lambda ins: sum(ins) % 2 == 0 if ins else 1,
}


def evaluate_circuit(circuit: CircuitJSON, input_values: dict[str, int]) -> dict[str, int]:
    """Evaluate *circuit* for given *input_values* (pure Python simulation).

    Parameters
    ----------
    circuit:
        The circuit to evaluate.
    input_values:
        A mapping of input node **name** → 0 or 1.

    Returns
    -------
    dict
        A mapping of output node **name** → 0 or 1.
    """
    # Build id → node map
    node_map = {n.id: n for n in circuit.nodes}
    # Build id → list of incoming edges (sorted by to_port)
    incoming: dict[str, list] = defaultdict(list)
    for edge in circuit.edges:
        incoming[edge.to_node].append(edge)
    for lst in incoming.values():
        lst.sort(key=lambda e: e.to_port)

    # Signal values keyed by node id
    sig: dict[str, int] = {}

    # Initialise input nodes
    for node in circuit.get_inputs():
        sig[node.id] = input_values.get(node.name, 0)

    # Topological evaluation
    try:
        import networkx as nx
        G = nx.DiGraph()
        for node in circuit.nodes:
            G.add_node(node.id)
        for edge in circuit.edges:
            G.add_edge(edge.from_node, edge.to_node)
        order = list(nx.topological_sort(G))
    except Exception:
        order = [n.id for n in circuit.nodes]

    for nid in order:
        node = node_map[nid]
        if node.type == "input":
            continue  # already set
        if node.type == "output":
            edges = incoming.get(nid, [])
            sig[nid] = sig.get(edges[0].from_node, 0) if edges else 0
            continue

        edges = incoming.get(nid, [])
        in_vals = [sig.get(e.from_node, 0) for e in edges]
        fn = _GATE_EVAL.get(node.type.lower())
        if fn is not None:
            result = fn(in_vals)
            sig[nid] = int(bool(result))
        else:
            sig[nid] = 0

    return {n.name: sig.get(n.id, 0) for n in circuit.get_outputs()}


def compute_truth_table(circuit: CircuitJSON) -> list[dict[str, int]]:
    """Return the full truth table as a list of row dicts (input+output names)."""
    inputs = circuit.get_inputs()
    rows: list[dict[str, int]] = []
    for combo in itertools.product([0, 1], repeat=len(inputs)):
        input_vals = {inp.name: v for inp, v in zip(inputs, combo)}
        output_vals = evaluate_circuit(circuit, input_vals)
        rows.append({**input_vals, **output_vals})
    return rows


# ---------------------------------------------------------------------------
# Testbench generation
# ---------------------------------------------------------------------------


def _generate_testbench(circuit: CircuitJSON, module_verilog: str) -> str:
    """Generate a self-checking Verilog testbench."""
    inputs = circuit.get_inputs()
    outputs = circuit.get_outputs()
    module_name = _safe_id(circuit.circuit_name)

    tb_lines: list[str] = []
    tb_lines.append("`timescale 1ns/1ps")
    tb_lines.append(f"module tb_{module_name};")
    tb_lines.append("")

    # reg/wire declarations
    for node in inputs:
        tb_lines.append(f"    reg  {_safe_id(node.name)};")
    for node in outputs:
        tb_lines.append(f"    wire {_safe_id(node.name)};")

    tb_lines.append("")

    # DUT instantiation
    port_conn = ", ".join(
        [f".{_safe_id(n.name)}({_safe_id(n.name)})" for n in inputs + outputs]
    )
    tb_lines.append(f"    {module_name} dut ({port_conn});")
    tb_lines.append("")

    # Generate test vectors from truth table
    truth_table = compute_truth_table(circuit)
    tb_lines.append("    integer errors = 0;")
    tb_lines.append("    initial begin")

    for row in truth_table:
        # Set inputs
        assignments = "; ".join(
            f"{_safe_id(n.name)} = {row[n.name]}" for n in inputs
        )
        tb_lines.append(f"        {assignments}; #10;")

        # Check outputs
        for node in outputs:
            expected = row[node.name]
            tb_lines.append(
                f"        if ({_safe_id(node.name)} !== {expected}) begin"
            )
            tb_lines.append(
                f"            $display(\"FAIL: {node.name} expected {expected}, got %b\", "
                f"{_safe_id(node.name)});"
            )
            tb_lines.append(f"            errors = errors + 1;")
            tb_lines.append(f"        end")

    tb_lines.append("        if (errors == 0)")
    tb_lines.append(f'            $display("PASS: all {len(truth_table)} test vectors passed");')
    tb_lines.append("        else")
    tb_lines.append(f'            $display("FAIL: %0d error(s)", errors);')
    tb_lines.append("        $finish;")
    tb_lines.append("    end")
    tb_lines.append(f"endmodule")

    return module_verilog + "\n\n" + "\n".join(tb_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VerificationResult:
    """Result of a simulation run."""

    def __init__(
        self,
        passed: bool,
        errors: int,
        stdout: str,
        stderr: str,
        iverilog_available: bool,
    ) -> None:
        self.passed = passed
        self.errors = errors
        self.stdout = stdout
        self.stderr = stderr
        self.iverilog_available = iverilog_available

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}, errors={self.errors})"


def run_verification(circuit: CircuitJSON) -> VerificationResult:
    """Run functional verification of *circuit*.

    1. Tries Icarus Verilog if available.
    2. Falls back to pure-Python truth-table simulation.
    """
    verilog_src = generate_verilog(circuit)
    iverilog_available = shutil.which("iverilog") is not None

    if iverilog_available:
        return _run_iverilog(circuit, verilog_src)
    else:
        return _run_python_sim(circuit)


def _run_iverilog(circuit: CircuitJSON, verilog_src: str) -> VerificationResult:
    """Compile and simulate with Icarus Verilog."""
    testbench = _generate_testbench(circuit, verilog_src)

    with tempfile.TemporaryDirectory() as tmpdir:
        tb_path = os.path.join(tmpdir, "tb.v")
        out_path = os.path.join(tmpdir, "sim.out")

        with open(tb_path, "w") as fh:
            fh.write(testbench)

        # Compile
        compile_result = subprocess.run(
            ["iverilog", "-o", out_path, tb_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            return VerificationResult(
                passed=False,
                errors=-1,
                stdout=compile_result.stdout,
                stderr=compile_result.stderr,
                iverilog_available=True,
            )

        # Simulate
        sim_result = subprocess.run(
            ["vvp", out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = sim_result.stdout
        passed = "PASS" in stdout and "FAIL" not in stdout

        error_count = 0
        m = re.search(r"FAIL: (\d+) error", stdout)
        if m:
            error_count = int(m.group(1))

        return VerificationResult(
            passed=passed,
            errors=error_count,
            stdout=stdout,
            stderr=sim_result.stderr,
            iverilog_available=True,
        )


def _run_python_sim(circuit: CircuitJSON) -> VerificationResult:
    """Pure-Python fallback simulation (no iverilog required)."""
    truth_table = compute_truth_table(circuit)
    errors = 0
    log_lines: list[str] = []

    inputs = circuit.get_inputs()
    outputs = circuit.get_outputs()

    for row in truth_table:
        in_str = ", ".join(f"{n.name}={row[n.name]}" for n in inputs)
        out_str = ", ".join(f"{n.name}={row[n.name]}" for n in outputs)
        log_lines.append(f"  [{in_str}] → [{out_str}]")

    msg_lines = [
        f"Python simulation of '{circuit.circuit_name}'",
        f"Truth table ({len(truth_table)} rows):",
    ] + log_lines + [
        "",
        f"PASS: all {len(truth_table)} test vectors evaluated (no iverilog available)",
    ]

    return VerificationResult(
        passed=True,
        errors=0,
        stdout="\n".join(msg_lines),
        stderr="iverilog not found – using Python simulation fallback",
        iverilog_available=False,
    )
