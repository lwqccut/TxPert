"""Precompute LLM features for selected, high-signal perturbation genes.

This package deliberately reuses the data and LLM/KG implementations from
``LLM_agent``.  It does not call the LLM for every gene in the expression
matrix.  Instead it selects a bounded set of perturbation genes using only
the existing TxPert data, then summarizes the existing STRING/GO evidence
for those genes and stores a dense feature table indexed by perturbation id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gspp.data.datamodule import PertDataModule
from gspp.data.graphmodule import GSPGraph
from LLM_agent.knowledge import KnowledgeGraph
from LLM_agent.llm_client import QwenLocalClient


def _as_dense_mean(matrix: Any) -> np.ndarray:
    value = matrix.mean(axis=0)
    if hasattr(value, "A1"):
        value = value.A1
    return np.asarray(value).reshape(-1).astype(np.float32)


def _row_abs_mean(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "multiply"):
        return np.asarray(np.abs(matrix).mean(axis=0)).reshape(-1).astype(np.float32)
    return np.asarray(np.abs(matrix).mean(axis=0)).reshape(-1).astype(np.float32)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def select_key_genes(dm: PertDataModule, selection_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank perturbation genes from existing TxPert data only.

    The score is deliberately data-driven: the perturbation gene's observed
    response signal, its existing Pharos level, and its occurrence frequency
    are combined.  No test labels or new corpus is introduced.
    """
    max_key_genes = max(1, int(selection_cfg.get("max_key_genes", 256)))
    response_weight = float(selection_cfg.get("response_weight", 0.60))
    pharos_weight = float(selection_cfg.get("pharos_weight", 0.25))
    frequency_weight = float(selection_cfg.get("frequency_weight", 0.15))

    gene_to_col = {str(name).upper(): idx for idx, name in enumerate(dm.adata.var_names)}
    # Key-gene selection must be train-only.  The test perturbations and test
    # controls are deliberately excluded to avoid leaking evaluation signals.
    treatment_mean = _as_dense_mean(dm.train_adata.X)
    control_mean = _as_dense_mean(dm.control_data_train.X)
    response_signal = _normalise(np.abs(treatment_mean - control_mean))
    pharos = np.asarray(dm.adata.uns.get("pharos", np.zeros(dm.adata.n_vars)), dtype=np.float32)
    pharos = _normalise(np.maximum(pharos, 0.0))

    labels = dm.train_adata.obs["condition"].astype(str).str.split("+").explode()
    frequency_map = labels.value_counts(normalize=True).to_dict()
    candidates: list[dict[str, Any]] = []
    for raw_id, label in dm.id2pert.items():
        index = int(raw_id)
        if index < 0 or str(label).lower() in {"ctrl", "control"}:
            continue
        gene = str(label).replace("+ctrl", "").upper()
        column = gene_to_col.get(gene)
        if column is None:
            continue
        frequency = float(frequency_map.get(str(label), frequency_map.get(gene, 0.0)))
        score = (
            response_weight * float(response_signal[column])
            + pharos_weight * float(pharos[column])
            + frequency_weight * frequency
        )
        candidates.append({
            "gene": gene,
            "label": str(label),
            "perturbation_id": index,
            "score": score,
            "response_signal": float(response_signal[column]),
            "pharos_score": float(pharos[column]),
            "frequency": frequency,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_key_genes]


def _forward_graph_edges(dm: PertDataModule) -> tuple[set[tuple[int, int]], dict[str, int]]:
    """Return edges retained by the current STRING/GO forward views.

    The forward model reduces both resources to perturbation nodes and keeps
    top-20 edges per target.  This set lets the LLM prompt distinguish
    evidence already represented in the model from evidence that the model's
    graph construction may have dropped.
    """
    graph = GSPGraph(dm.pert2id, dm.gene2id, {
        name: {"graph_type": name, "reduce2perts": True, "mode": "top_20"}
        for name in ("string", "go")
    })
    edges: set[tuple[int, int]] = set()
    for edge_index, _, _ in graph.graph_dict.values():
        edges.update(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    label_to_id = {
        str(label).replace("+ctrl", "").upper(): int(index)
        for label, index in dm.pert2id.items()
        if int(index) >= 0
    }
    return edges, label_to_id


def build_features(config: dict[str, Any], output: Path) -> None:
    retrieval = config.get("retrieval", {})
    model_cfg = config.get("model", {})
    dm_cfg = dict(config.get("forward", {}).get("datamodule", {}))
    dm_cfg.setdefault("mode", "inference")
    dm_cfg.setdefault("task_type", "K562_single_cell_line")
    dm_cfg.setdefault("obsm_key", "raw")
    dm = PertDataModule(**dm_cfg)
    dm.prepare_data()
    dm.setup("fit")

    selected = select_key_genes(dm, config.get("key_gene_selection", {}))
    forward_edges, perturbation_ids = _forward_graph_edges(dm)
    kg = KnowledgeGraph(retrieval.get("kg_paths", []))
    qwen = QwenLocalClient(
        model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct"),
        model_cfg.get("cache_dir"), model_cfg.get("device", "auto"),
        int(model_cfg.get("max_new_tokens", 512)),
        float(model_cfg.get("temperature", 0.0)),
        bool(model_cfg.get("auto_install", True)), str(model_cfg.get("dtype", "float16")),
    )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers before precomputing key-gene features") from exc
    encoder = SentenceTransformer(
        retrieval.get("embedding_model", "BAAI/bge-m3"),
        cache_folder=model_cfg.get("cache_dir"),
    )

    num_nodes = max(int(key) for key in dm.id2pert if int(key) >= 0) + 1
    texts: list[str] = []
    selected_ids: list[int] = []
    graph_analysis: list[dict[str, Any]] = []
    omitted_evidence: list[dict[str, Any]] = []
    top_k = int(retrieval.get("top_k", 8))
    for item in selected:
        gene = item["gene"]
        evidence = kg.query([gene], top_k)
        evidence_lines = []
        gene_records = []
        for entry in evidence:
            source = str(entry.metadata.get("source", "")).upper()
            target = str(entry.metadata.get("target", "")).upper()
            source_id = perturbation_ids.get(source)
            target_id = perturbation_ids.get(target)
            represented = (
                source_id is not None and target_id is not None
                and (source_id, target_id) in forward_edges
            )
            record = {
                "gene": gene, "source": source, "target": target,
                "database": entry.metadata.get("database", entry.source),
                "score": float(entry.score),
                "represented_in_forward_graph": represented,
                "source_is_perturbation_node": source_id is not None,
                "target_is_perturbation_node": target_id is not None,
            }
            gene_records.append(record)
            evidence_lines.append(
                f"[{record['database']}; "
                f"{'already represented' if represented else 'candidate omitted from forward graph'}] "
                f"{entry.text}"
            )
            if not represented:
                omitted_evidence.append(record)
        evidence_text = "\n".join(evidence_lines)
        graph_analysis.append({"gene": gene, "evidence": gene_records})
        prompt = (
            f"Key perturbation gene: {gene}\n"
            f"Cell type: {dm.test_cell_type}\n"
            f"Observed dataset signal: {item['response_signal']:.4f}\n"
            "The forward model currently reduces STRING and GO to perturbation "
            "nodes and retains only top-20 edges per target. Evidence marked "
            "'candidate omitted from forward graph' may contain information the "
            "forward graph does not represent.\n"
            f"Evidence from existing TxPert knowledge graphs:\n{evidence_text}\n"
            "Summarize only evidence-backed key gene associations, shared "
            "functional context, and perturbation-relevant relationships. Give "
            "special attention to high-confidence candidate omitted relations, "
            "but do not invent mechanisms or numerical expression values."
        )
        analysis = qwen.complete(
            "You are a cautious genomics knowledge analyst. Use only the supplied evidence.",
            prompt,
        )
        texts.append(f"Key gene: {gene}\n{analysis}\n{evidence_text}")
        selected_ids.append(int(item["perturbation_id"]))

    selected_embeddings = encoder.encode(
        texts, normalize_embeddings=True, convert_to_tensor=True,
    ).cpu() if texts else torch.zeros((0, encoder.get_sentence_embedding_dimension()))
    embedding_dim = int(selected_embeddings.shape[1])
    embeddings = torch.zeros((num_nodes, embedding_dim), dtype=torch.float32)
    key_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if selected_ids:
        embeddings[selected_ids] = selected_embeddings
        key_mask[selected_ids] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "key_mask": key_mask,
        "selected_genes": selected,
        "graph_analysis": graph_analysis,
        "omitted_evidence": omitted_evidence,
        "forward_graph_policy": "STRING+GO reduced to perturbation nodes, top_20 per target",
        "gene_to_id": {item["gene"]: item["perturbation_id"] for item in selected},
        "pert_to_id": {str(key): int(value) for key, value in dm.pert2id.items()},
        "num_nodes": num_nodes,
        "embedding_model": retrieval.get("embedding_model", "BAAI/bge-m3"),
        "source_paths": retrieval.get("kg_paths", []),
    }, output)
    print(json.dumps({
        "output": str(output), "num_nodes": num_nodes,
        "selected_genes": len(selected), "embedding_dim": embedding_dim,
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("LLM_agent_plus/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("LLM_agent_plus/runs/key_gene_features.pt"))
    args = parser.parse_args()
    import yaml
    build_features(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()

