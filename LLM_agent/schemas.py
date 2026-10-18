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


@dataclass(frozen=True)
class AgentPlan:
    genes: list[str]
    perturbation_type: str
    selected_evidence: list[str]
    confidence: float
    rationale: str


def parse_agent_plan(raw: str, valid_genes: Sequence[str], evidence_ids: set[str]) -> AgentPlan:
    """Parse and validate Qwen's plan; never accept invented genes or evidence."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen did not return a JSON agent plan")
    try:
        item = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen returned invalid JSON for the agent plan") from exc
    genes = [str(g).upper() for g in item.get("genes", [])]
    allowed = {str(g).upper() for g in valid_genes}
    if not genes or not set(genes).issubset(allowed):
        raise ValueError("Qwen agent plan contains genes outside the resolved task")
    selected = [str(value) for value in item.get("selected_evidence", [])]
    if not set(selected).issubset(evidence_ids):
        raise ValueError("Qwen agent plan references evidence that was not retrieved")
    confidence = float(item.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Qwen confidence must be between 0 and 1")
    return AgentPlan(genes, str(item.get("perturbation_type", "unspecified")), selected,
                     confidence, str(item.get("rationale", "")))


def parse_task(data: Mapping[str, Any]) -> GeneTask:
    task = GeneTask(**{key: value for key, value in data.items() if key in GeneTask.__dataclass_fields__})
    if isinstance(task.genes, str):
        task.genes = [item.strip() for item in task.genes.replace(",", " ").split() if item.strip()]
    task.perturbation_ids = [int(item) for item in task.perturbation_ids]
    task.control_expression = [float(item) for item in task.control_expression]
    return task
