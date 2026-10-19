"""Precompute real gene knowledge vectors for knowledge-augmented training.

This script reads the existing TxPert gene set and KG files.  Qwen summarizes
retrieved evidence for each perturbation and BGE-M3 turns that analysis into a
fixed vector stored in a torch artifact.  No synthetic or example records are
created.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch

from gspp.data.datamodule import PertDataModule
from .knowledge import KnowledgeGraph
from .llm_client import QwenLocalClient


def build_features(config: dict, output: Path) -> None:
    retrieval = config.get("retrieval", {})
    model_cfg = config.get("model", {})
    dm_cfg = dict(config.get("forward", {}).get("datamodule", {}))
    dm_cfg.setdefault("mode", "inference")
    dm_cfg.setdefault("task_type", "K562_single_cell_line")
    dm_cfg.setdefault("obsm_key", "raw")
    dm = PertDataModule(**dm_cfg)
    dm.prepare_data()
    kg = KnowledgeGraph(retrieval.get("kg_paths", []))
    qwen = QwenLocalClient(
        model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct"), model_cfg.get("cache_dir"),
        model_cfg.get("device", "auto"), int(model_cfg.get("max_new_tokens", 512)),
        float(model_cfg.get("temperature", 0.0)), bool(model_cfg.get("auto_install", True)),
        str(model_cfg.get("dtype", "float16")),
    )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers before precomputing knowledge") from exc
    encoder = SentenceTransformer(retrieval.get("embedding_model", "BAAI/bge-m3"), cache_folder=model_cfg.get("cache_dir"))

    num_nodes = max(int(key) for key in dm.id2pert if int(key) >= 0) + 1
    texts: list[str] = [""] * num_nodes
    ids: dict[str, int] = {}
    top_k = int(retrieval.get("top_k", 8))
    for raw_id, label in dm.id2pert.items():
        index = int(raw_id)
        if index < 0 or str(label).lower() in {"ctrl", "control"}:
            continue
        gene = str(label).replace("+ctrl", "").upper()
        evidence = kg.query([gene], top_k)
        evidence_text = "\n".join(item.text for item in evidence)
        prompt = (f"Gene perturbation: {gene}\nCell type: {dm.test_cell_type}\n"
                  f"Evidence from TxPert knowledge graphs:\n{evidence_text}\n"
                  "Summarize only evidence-backed gene associations and perturbation context. "
                  "Do not invent mechanisms or numerical expression values.")
        analysis = qwen.complete(
            "You are a genomics knowledge analyst. Do not make unsupported claims.", prompt
        )
        texts[index] = f"Gene: {gene}\n{analysis}\n{evidence_text}"
        ids[gene] = index
    embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings.cpu(), "gene_to_id": ids,
        "pert_to_id": {str(key): int(value) for key, value in dm.pert2id.items()},
        "num_nodes": num_nodes, "embedding_model": retrieval.get("embedding_model", "BAAI/bge-m3"),
        "source_paths": retrieval.get("kg_paths", []),
    }, output)
    print(json.dumps({"output": str(output), "num_nodes": num_nodes,
                      "embedding_dim": int(embeddings.shape[1])}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("LLM_agent/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("LLM_agent/runs/knowledge_features.pt"))
    args = parser.parse_args()
    import yaml
    build_features(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
