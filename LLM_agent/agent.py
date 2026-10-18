"""LLM orchestration kept separate from TxPert's training scripts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable, Sequence

from .evidence import Evidence, JsonlEvidenceStore, evidence_to_dict
from .llm_client import LLMClient


@dataclass(frozen=True)
class AgentRequest:
    control_expression: Sequence[float]
    perturbation_ids: Sequence[int]
    perturbation_name: str = ""
    cell_type: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    prediction: Any
    selected_model: str
    rationale: str
    kg_evidence: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class LLMAgent:
    """Select a model, summarize evidence and call the numerical predictor."""

    def __init__(self, forward_model: Callable[..., Any], *, kg: JsonlEvidenceStore | None = None,
                 rag: JsonlEvidenceStore | None = None, llm: LLMClient | None = None,
                 max_evidence: int = 8):
        self.forward_model = forward_model
        self.kg = kg
        self.rag = rag
        self.llm = llm or LLMClient()
        self.max_evidence = max_evidence

    def run(self, request: AgentRequest) -> AgentResponse:
        self._validate(request)
        query = " ".join(filter(None, [request.perturbation_name, request.cell_type, *map(str, request.perturbation_ids)]))
        kg_items = self.kg.search(query, self.max_evidence) if self.kg else []
        rag_items = self.rag.search(query, self.max_evidence) if self.rag else []
        evidence = kg_items + rag_items
        prompt = self._prompt(request, evidence)
        rationale = self.llm.complete("You are a cautious scientific model router.", prompt)
        prediction = self.forward_model(request.control_expression, request.perturbation_ids, {
            **request.context, "evidence": evidence, "rationale": rationale,
        })
        return AgentResponse(prediction, "txpert", rationale, evidence_to_dict(kg_items), evidence_to_dict(rag_items))

    @staticmethod
    def _prompt(request: AgentRequest, evidence: Sequence[Evidence]) -> str:
        return json.dumps({"task": "predict perturbation effect", "cell_type": request.cell_type,
                           "perturbation_ids": request.perturbation_ids,
                           "evidence": evidence_to_dict(evidence)}, ensure_ascii=False)

    @staticmethod
    def _validate(request: AgentRequest) -> None:
        if not request.control_expression:
            raise ValueError("control_expression must not be empty")
        if not request.perturbation_ids:
            raise ValueError("perturbation_ids must not be empty")
        if any(not isinstance(item, int) or item < 0 for item in request.perturbation_ids):
            raise ValueError("perturbation_ids must contain non-negative integers")
