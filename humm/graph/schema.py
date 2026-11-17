"""Humm Causal Graph (HCG) schema."""
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class EvidenceKind(str, Enum):
    TEXT = "text"
    DATA = "data"
    LLM_PRIOR = "llm_prior"
    EXPERT = "expert"
    EXPERIMENT = "experiment"


class EdgeStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    REFUTED = "refuted"


# Weight per evidence kind: experiment > data ~ expert > text > LLM prior
KIND_WEIGHT = {
    EvidenceKind.EXPERIMENT: 1.0,
    EvidenceKind.DATA: 0.8,
    EvidenceKind.EXPERT: 0.8,
    EvidenceKind.TEXT: 0.6,
    EvidenceKind.LLM_PRIOR: 0.4,
}


class Source(BaseModel):
    kind: Literal["text", "table", "llm", "expert"]
    ref: str


class Node(BaseModel):
    id: str
    label: str = ""
    type: Literal["variable", "event"] = "variable"
    dtype: Literal["binary", "categorical", "continuous", "unknown"] = "unknown"
    aliases: list[str] = []
    source: list[Source] = []


class Evidence(BaseModel):
    kind: EvidenceKind
    score: float = Field(ge=0, le=1)
    ref: str = ""
    span: str = ""
    method: str = ""


class Edge(BaseModel):
    src: str
    dst: str
    sign: Literal["+", "-", "0", "?"] = "?"
    lag: int = 0
    mechanism: str = ""
    evidence: list[Evidence] = []
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: EdgeStatus = EdgeStatus.HYPOTHESIS

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)

    def update_confidence(self) -> float:
        """noisy-OR combination: 1 - prod(1 - w_kind * score)."""
        miss = 1.0
        for e in self.evidence:
            miss *= 1 - KIND_WEIGHT[e.kind] * e.score
        self.confidence = 1 - miss
        return self.confidence

    def promote(self) -> EdgeStatus:
        """hypothesis -> supported: when there are at least 2 distinct kinds of evidence. verified/refuted decisions are made by humm.validate."""
        kinds = {e.kind for e in self.evidence}
        self.status = EdgeStatus.SUPPORTED if self.status == EdgeStatus.HYPOTHESIS and len(kinds) >= 2 else self.status
        return self.status


class Provenance(BaseModel):
    created: date = Field(default_factory=date.today)
    pipeline: str = "humm 0.0.1"


class HCG(BaseModel):
    graph_id: str
    nodes: list[Node] = []
    edges: list[Edge] = []
    assumptions: list[str] = []
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _check(self):
        ids = [n.id for n in self.nodes]
        assert len(ids) == len(set(ids)), "duplicate node ids"
        unknown = {x for e in self.edges for x in e.key} - set(ids)
        assert not unknown, f"edges reference undefined nodes: {sorted(unknown)}"
        keys = [e.key for e in self.edges]
        assert len(keys) == len(set(keys)), "duplicate edges"
        return self

    # --- queries ---
    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)

    def edge(self, src: str, dst: str) -> Edge:
        return next(e for e in self.edges if e.key == (src, dst))

    def active_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.status != EdgeStatus.REFUTED]

    # --- mutations ---
    def add_node(self, node_id: str, **kw) -> Node:
        node = Node(id=node_id, **kw)
        self.nodes.append(node)
        return node

    def add_edge(self, src: str, dst: str, **kw) -> Edge:
        edge = Edge(src=src, dst=dst, **kw)
        edge.update_confidence()
        self.edges.append(edge)
        self._check()
        return edge

    # --- graph operations ---
    def to_networkx(self, include_refuted: bool = False) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from((n.id, n.model_dump(exclude={"id"})) for n in self.nodes)
        edges = self.edges if include_refuted else self.active_edges()
        g.add_edges_from((e.src, e.dst, e.model_dump(exclude={"src", "dst"})) for e in edges)
        return g

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.to_networkx())

    def cycles(self) -> list[list[str]]:
        return list(nx.simple_cycles(self.to_networkx()))

    def break_cycles(self) -> list[tuple[str, str]]:
        """For each cycle, mark the lowest-confidence edge as refuted."""
        refuted = []
        while not self.is_dag():
            cycle = self.cycles()[0]
            pairs = list(zip(cycle, cycle[1:] + cycle[:1]))
            weakest = min((self.edge(*p) for p in pairs), key=lambda e: e.confidence)
            weakest.status = EdgeStatus.REFUTED
            refuted.append(weakest.key)
        return refuted

    @classmethod
    def from_networkx(cls, g: nx.DiGraph, graph_id: str, **kw) -> "HCG":
        nodes = [Node(id=str(n)) for n in g.nodes]
        edges = [Edge(src=str(u), dst=str(v)) for u, v in g.edges]
        return cls(graph_id=graph_id, nodes=nodes, edges=edges, **kw)

    # --- persistence ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HCG":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
