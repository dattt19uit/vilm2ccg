"""
circuit_json.py – CircuitJSON data structure.

CircuitJSON is the intermediate representation used throughout ViLM2CCG.
It describes a combinational circuit as a directed graph with:
  - nodes: gates (AND, OR, NOT, …) and I/O ports
  - edges: wires connecting node ports

Serialization format
--------------------
::

    {
      "id": "<circuit-id>",
      "input_text": "<Vietnamese NL description>",
      "graph": {
        "nodes": [{"id": "...", "type": "...", "name": "...", ...}],
        "edges": [{"source": "...", "target": "...", "name": "...", ...}]
      },
      "verilog": "<Verilog source>",
      "metadata": {
        "num_inputs": <int>,
        "num_outputs": <int>,
        "num_gates": <int>,
        "circuit_name": "...",
        "circuit_type": "...",
        "description": "..."
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node and edge types
# ---------------------------------------------------------------------------

VALID_NODE_TYPES = {
    "input",
    "output",
    "and",
    "or",
    "not",
    "nand",
    "nor",
    "xor",
    "xnor",
    "buf",
    "mux",
    "wire",
}


@dataclass
class CircuitNode:
    """A single node (gate or I/O port) in the circuit."""

    id: str
    type: str  # one of VALID_NODE_TYPES
    name: str
    width: int = 1
    num_inputs: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "width": self.width,
            "num_inputs": self.num_inputs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitNode":
        return cls(
            id=d["id"],
            type=d["type"],
            name=d.get("name", d["id"]),
            width=d.get("width", 1),
            num_inputs=d.get("num_inputs", 2),
            metadata=d.get("metadata", {}),
        )


@dataclass
class CircuitEdge:
    """A directed wire connecting two node ports."""

    from_node: str   # source node id
    to_node: str     # destination node id
    from_port: int = 0  # output port index of source
    to_port: int = 0    # input port index of destination
    name: str = ""   # optional net name

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.from_node,
            "target": self.to_node,
            "from_port": self.from_port,
            "to_port": self.to_port,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitEdge":
        # Accept both new-format keys ("source"/"target") and
        # legacy keys ("from"/"to") for backward compatibility.
        from_node = d.get("source") or d.get("from", "")
        to_node = d.get("target") or d.get("to", "")
        return cls(
            from_node=from_node,
            to_node=to_node,
            from_port=d.get("from_port", 0),
            to_port=d.get("to_port", 0),
            name=d.get("name", ""),
        )


# ---------------------------------------------------------------------------
# CircuitJSON container
# ---------------------------------------------------------------------------


class CircuitJSON:
    """Container for a complete combinational circuit description."""

    def __init__(
        self,
        circuit_name: str = "circuit",
        description: str = "",
        circuit_type: str = "custom",
        id: str = "",
        input_text: str = "",
        verilog: str = "",
    ) -> None:
        self.circuit_name = circuit_name
        self.description = description
        self.circuit_type = circuit_type  # e.g. "and", "mux", "half_adder", …
        # Public-facing ID and NL description (used in dataset / serialization)
        self.id = id or circuit_name
        self.input_text = input_text or description
        # Optional embedded Verilog (populated by hdl_generator when needed)
        self.verilog = verilog
        self.nodes: list[CircuitNode] = []
        self.edges: list[CircuitEdge] = []

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def add_node(self, node: CircuitNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: CircuitEdge) -> None:
        self.edges.append(edge)

    def get_inputs(self) -> list[CircuitNode]:
        return [n for n in self.nodes if n.type == "input"]

    def get_outputs(self) -> list[CircuitNode]:
        return [n for n in self.nodes if n.type == "output"]

    def get_gates(self) -> list[CircuitNode]:
        return [n for n in self.nodes if n.type not in {"input", "output"}]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical CircuitJSON schema."""
        return {
            "id": self.id or self.circuit_name,
            "input_text": self.input_text or self.description,
            "graph": {
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
            },
            "verilog": self.verilog,
            "metadata": {
                "num_inputs": len(self.get_inputs()),
                "num_outputs": len(self.get_outputs()),
                "num_gates": len(self.get_gates()),
                "circuit_name": self.circuit_name,
                "circuit_type": self.circuit_type,
                "description": self.description,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitJSON":
        """Deserialise from the canonical schema or the legacy schema."""
        if "graph" in d:
            # ── New canonical format ──────────────────────────────────
            meta = d.get("metadata", {})
            cj = cls(
                circuit_name=meta.get("circuit_name", d.get("id", "circuit")),
                description=meta.get("description", d.get("input_text", "")),
                circuit_type=meta.get("circuit_type", "custom"),
                id=d.get("id", ""),
                input_text=d.get("input_text", ""),
                verilog=d.get("verilog", ""),
            )
            graph = d.get("graph", {})
            for nd in graph.get("nodes", []):
                cj.add_node(CircuitNode.from_dict(nd))
            for ed in graph.get("edges", []):
                cj.add_edge(CircuitEdge.from_dict(ed))
        else:
            # ── Legacy format (circuit_name / nodes / edges at top level) ──
            cj = cls(
                circuit_name=d.get("circuit_name", "circuit"),
                description=d.get("description", ""),
                circuit_type=d.get("circuit_type", "custom"),
            )
            for nd in d.get("nodes", []):
                cj.add_node(CircuitNode.from_dict(nd))
            for ed in d.get("edges", []):
                cj.add_edge(CircuitEdge.from_dict(ed))
        return cj

    @classmethod
    def from_json(cls, json_str: str) -> "CircuitJSON":
        return cls.from_dict(json.loads(json_str))
