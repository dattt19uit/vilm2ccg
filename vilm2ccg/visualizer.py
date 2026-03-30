"""
visualizer.py – CCG visualization using NetworkX and Matplotlib.

Generates static PNG/SVG graphs of CircuitJSON circuits.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from vilm2ccg.circuit_json import CircuitJSON


# ---------------------------------------------------------------------------
# Node colour/shape mapping
# ---------------------------------------------------------------------------

_NODE_COLORS: dict[str, str] = {
    "input": "#4CAF50",   # green
    "output": "#F44336",  # red
    "and": "#2196F3",     # blue
    "or": "#9C27B0",      # purple
    "not": "#FF9800",     # orange
    "buf": "#FFEB3B",     # yellow
    "nand": "#00BCD4",    # cyan
    "nor": "#E91E63",     # pink
    "xor": "#3F51B5",     # indigo
    "xnor": "#009688",    # teal
    "mux": "#795548",     # brown
    "wire": "#607D8B",    # blue-grey
}

_DEFAULT_COLOR = "#9E9E9E"  # grey


def _get_node_color(node_type: str) -> str:
    return _NODE_COLORS.get(node_type.lower(), _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_nx_graph(circuit: CircuitJSON) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in circuit.nodes:
        G.add_node(
            node.id,
            label=f"{node.name}\n({node.type})",
            node_type=node.type,
            color=_get_node_color(node.type),
        )

    for edge in circuit.edges:
        label = edge.name or ""
        G.add_edge(edge.from_node, edge.to_node, label=label)

    return G


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _layered_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Compute a left-to-right layered layout using topological sort."""
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G, seed=42)

    # Assign layer (longest path from source)
    layer: dict[str, int] = {}
    for node in topo:
        preds = list(G.predecessors(node))
        layer[node] = (max(layer[p] for p in preds) + 1) if preds else 0

    # Group by layer
    by_layer: dict[int, list[str]] = {}
    for node, lyr in layer.items():
        by_layer.setdefault(lyr, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for lyr, nodes in by_layer.items():
        for i, node in enumerate(nodes):
            pos[node] = (lyr * 2.5, -i * 1.8)

    return pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_circuit(
    circuit: CircuitJSON,
    output_path: str | None = None,
    fmt: str = "png",
    figsize: tuple[float, float] = (14, 8),
    show_edge_labels: bool = True,
) -> bytes:
    """Render *circuit* as a graph image.

    Parameters
    ----------
    circuit:
        The :class:`~vilm2ccg.circuit_json.CircuitJSON` to render.
    output_path:
        If given, save the image to this path.
    fmt:
        Image format: ``"png"`` (default) or ``"svg"``.
    figsize:
        Matplotlib figure size in inches.
    show_edge_labels:
        If ``True``, annotate edges with their net name.

    Returns
    -------
    bytes
        The raw image bytes (PNG or SVG).
    """
    G = _build_nx_graph(circuit)

    if len(G.nodes) == 0:
        # Empty graph – return a placeholder image
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Empty Circuit", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        pos = _layered_layout(G)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        colors = [G.nodes[n].get("color", _DEFAULT_COLOR) for n in G.nodes]
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes}

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1800, alpha=0.92)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_color="white",
                                font_weight="bold")
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            arrows=True, arrowstyle="-|>", arrowsize=20,
            edge_color="#555555", width=1.5,
            connectionstyle="arc3,rad=0.05",
        )

        if show_edge_labels:
            edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True) if d.get("label")}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6,
                                         font_color="#333333")

        # Legend
        legend_handles = [
            mpatches.Patch(color=color, label=ntype.upper())
            for ntype, color in sorted(_NODE_COLORS.items())
            if any(G.nodes[n].get("node_type") == ntype for n in G.nodes)
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.8)

        ax.set_title(
            f"Circuit: {circuit.circuit_name}\n{circuit.description}",
            fontsize=11, pad=12,
        )
        ax.axis("off")
        plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.read()

    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(image_bytes)

    return image_bytes
