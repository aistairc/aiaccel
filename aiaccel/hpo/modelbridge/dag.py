"""Lightweight DAG representation for pipeline planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import PhaseContext, RunnerFn


@dataclass(slots=True)
class DagNode:
    """Node in the execution DAG."""

    context: PhaseContext
    run: RunnerFn
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineDag:
    """DAG of PhaseContext execution."""

    nodes: dict[str, DagNode]

    def topological(self) -> list[DagNode]:
        """Return nodes in a stable topological order."""

        ordered: list[DagNode] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise RuntimeError("Cycle detected in pipeline DAG")
            temporary.add(name)
            node = self.nodes[name]
            for dep in node.depends_on:
                if dep in self.nodes:
                    visit(dep)
            permanent.add(name)
            ordered.append(node)

        for name in list(self.nodes):
            if name not in permanent:
                visit(name)
        return ordered


__all__ = ["PipelineDag", "DagNode"]
