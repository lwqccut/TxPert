"""End-to-end gene perturbation workflow."""
from __future__ import annotations

from typing import Any, Mapping

from .forward import build_forward_model
from .gene_processor import GeneProcessor
from .knowledge import KnowledgeGraph
from .llm_client import LLMClient, QwenLocalClient
from .rag import RAGIndex
from .schemas import AgentResult, GeneTask


class ResearchAgent:
    def __init__(self, config: Mapping[str, Any], *, processor: GeneProcessor | None = None):
        model_cfg = config.get("model", {})
        retrieval_cfg = config.get("retrieval", {})
        self.processor = processor or GeneProcessor()
        self.kg = KnowledgeGraph(retrieval_cfg.get("kg_path"))
        self.rag = RAGIndex(retrieval_cfg.get("rag_path"), retrieval_cfg.get("embedding_model", "BAAI/bge-m3"), model_cfg.get("cache_dir"))
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
        if not task.control_expression:
            raise ValueError("control_expression is required for numerical prediction")
        kg = self.kg.query(task.genes, self.top_k)
        query = " ".join([task.prompt, task.cell_type, *task.genes])
        rag = self.rag.search(query, self.top_k)
        evidence = kg + rag
        rationale = self.llm.complete(
            "You are a cautious genomics research agent. Explain evidence and route the numerical prediction.",
            f"Task: {task.prompt}\nGenes: {task.genes}\nEvidence: {[item.text for item in evidence]}",
        )
        prediction = self.forward(task.control_expression, task.perturbation_ids, {"evidence": evidence, "rationale": rationale})
        return AgentResult(task, prediction, "txpert-forward", rationale, evidence)
