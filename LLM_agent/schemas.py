"""Typed records passed between the agent modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import json


@dataclass
class GeneTask:
    prompt: str = ""
    cell_type: str = "K562"
    genes: list[str] = field(default_factory=list)
    perturbation_ids: list[int] = field(default_factory=list)
    control_expression: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str
    score: float
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    task: GeneTask
    prediction: Any
    selected_model: str
    rationale: str
    evidence: list[Evidence]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_task(data: Mapping[str, Any]) -> GeneTask:
    task = GeneTask(**{key: value for key, value in data.items() if key in GeneTask.__dataclass_fields__})
    if isinstance(task.genes, str):
        task.genes = [item.strip() for item in task.genes.replace(",", " ").split() if item.strip()]
    task.perturbation_ids = [int(item) for item in task.perturbation_ids]
    task.control_expression = [float(item) for item in task.control_expression]
    return task
