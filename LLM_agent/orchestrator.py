"""End-to-end gene perturbation workflow."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
from pathlib import Path

from .forward import build_forward_model
from .gene_processor import GeneProcessor
from .knowledge import KnowledgeGraph
from .llm_client import LLMClient, QwenLocalClient
from .rag import RAGIndex
from .schemas import AgentResult, GeneTask, parse_agent_plan


class ResearchAgent:
    def __init__(self, config: Mapping[str, Any], *, processor: GeneProcessor | None = None):
        model_cfg = config.get("model", {})
        retrieval_cfg = config.get("retrieval", {})
        project_root = Path(__file__).resolve().parents[1]
        resolve = lambda values: [str((project_root / value).resolve()) if not Path(value).is_absolute() else str(value) for value in values]
        self.processor = processor or GeneProcessor()
        self.kg = KnowledgeGraph(resolve(retrieval_cfg.get("kg_paths", [])))
        rag_paths = resolve(retrieval_cfg.get("rag_paths", []))
        self.rag = RAGIndex(rag_paths, retrieval_cfg.get("embedding_model", "BAAI/bge-m3"), model_cfg.get("cache_dir"))
        self.agent_cfg = config.get("agent", {})
        self.forward = build_forward_model(config.get("forward", {}))
        if model_cfg.get("enabled", False):
            self.llm = QwenLocalClient(
                model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct"),
                model_cfg.get("cache_dir"), model_cfg.get("device", "auto"),
                int(model_cfg.get("max_new_tokens", 768)), float(model_cfg.get("temperature", 0.0)),
                bool(model_cfg.get("auto_install", True)),
            )
        else:
            self.llm = LLMClient(base_url=model_cfg.get("base_url"), api_key=model_cfg.get("api_key"), model=model_cfg.get("name"))
        self.top_k = int(retrieval_cfg.get("top_k", 8))

    def run(self, task: GeneTask) -> AgentResult:
        if not task.genes and task.prompt:
            parsed = self.processor.extract(task.prompt, cell_type=task.cell_type)
            task.genes, task.perturbation_ids = parsed.genes, parsed.perturbation_ids
        if not task.perturbation_ids and hasattr(self.forward, "resolve_perturbations"):
            task.perturbation_ids = self.forward.resolve_perturbations(task.genes)
        if not task.control_expression and hasattr(self.forward, "default_control"):
            task.control_expression = self.forward.default_control()
        if not task.control_expression:
            raise ValueError("control_expression is required for numerical prediction")
        kg = self.kg.query(task.genes, self.top_k)
        query = " ".join([task.prompt, task.cell_type, *task.genes])
        rag = self.rag.search(query, self.top_k)
        evidence = kg + rag
        expression_profile = (
            self.forward.analysis_context(task.genes)
            if hasattr(self.forward, "analysis_context") else {}
        )
        evidence_payload = [{"id": f"E{index}", "kind": item.kind, "source": item.source,
                             "text": item.text, "score": item.score}
                            for index, item in enumerate(evidence)]
        mechanism_instruction = (
            "Do not explain mechanisms; return evidence-backed knowledge analysis only."
            if not self.agent_cfg.get("explain_mechanism", False)
            else "You may provide a clearly labelled mechanism hypothesis."
        )
        raw_plan = self.llm.complete(
            "You are a cautious genomics knowledge analyst. Return JSON only with keys: "
            "genes (a subset of the resolved genes), perturbation_type, selected_evidence "
            "(IDs only), confidence (0 to 1), rationale. Never invent genes, IDs, or evidence. "
            + mechanism_instruction,
            f"Task: {task.prompt}\nResolved genes: {task.genes}\n"
            f"Expression profile (selected genes and summary only): {expression_profile}\n"
            f"Evidence: {evidence_payload}",
        )
        plan = parse_agent_plan(raw_plan, task.genes, {item["id"] for item in evidence_payload})
        prediction = self.forward(task.control_expression, task.perturbation_ids, {
            "evidence": evidence, "plan": plan,
        })
        return AgentResult(task, prediction, "txpert-checkpoint", plan.rationale, evidence,
                           {"agent_confidence": plan.confidence})
