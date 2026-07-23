"""DAG helpers shared by the moderator (plan validation) and the orchestrator
(scheduling)."""
from __future__ import annotations

import networkx as nx

from moce.schema import Plan


class DagError(ValueError):
    """Raised when a Plan's block dependency graph is invalid."""


def build_graph(plan: Plan) -> nx.DiGraph:
    """Build a directed graph of block_id -> block_id (dependency -> dependent)."""
    graph = nx.DiGraph()
    ids = {b.id for b in plan.blocks}
    for block in plan.blocks:
        graph.add_node(block.id)
    for block in plan.blocks:
        for dep in block.depends_on:
            if dep not in ids:
                raise DagError(
                    f"block '{block.id}' depends on unknown block '{dep}'"
                )
            graph.add_edge(dep, block.id)
    return graph


def validate_dag(plan: Plan) -> nx.DiGraph:
    """Validate the plan's dependency graph, raising DagError if invalid.

    Returns the built graph on success (dependency -> dependent edges).
    """
    graph = build_graph(plan)
    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        raise DagError(f"plan contains cyclic block dependencies: {cycles}")
    return graph


def topological_generations(plan: Plan) -> list[list[str]]:
    """Return block ids grouped into ordered "generations" that can each run
    concurrently (every block in generation N only depends on blocks in
    generations < N)."""
    graph = validate_dag(plan)
    return [list(gen) for gen in nx.topological_generations(graph)]
