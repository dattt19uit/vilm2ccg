"""
circuit_json.py – CircuitJSON data structure.

CircuitJSON is the intermediate representation used throughout ViLM2CCG.
It describes a combinational circuit as a directed graph with:
  - nodes: gates (AND, OR, NOT, …) and I/O ports
  - edges: wires connecting node ports
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
            name=d["name"],
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
            "from": self.from_node,
            "to": self.to_node,
            "from_port": self.from_port,
            "to_port": self.to_port,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitEdge":
        return cls(
            from_node=d["from"],
            to_node=d["to"],
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
    ) -> None:
        self.circuit_name = circuit_name
        self.description = description
        self.circuit_type = circuit_type  # e.g. "and", "mux", "half_adder", …
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
        return {
            "circuit_name": self.circuit_name,
            "description": self.description,
            "circuit_type": self.circuit_type,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitJSON":
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
