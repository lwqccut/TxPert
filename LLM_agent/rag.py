"""BGE-M3 RAG index with a lexical fallback for CPU/offline smoke tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Evidence


class RAGIndex:
    def __init__(self, path: str | Path | None = None, embedding_model: str = "BAAI/bge-m3", cache_dir: str | None = None):
        self.documents: list[dict[str, Any]] = []
        self.encoder = None
        if path and Path(path).exists():
            self._load(Path(path))
        self._load_encoder(embedding_model, cache_dir)

    def _load(self, path: Path) -> None:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                self.documents = [json.loads(line) for line in handle if line.strip()]
        else:
            self.documents = [{"text": line.strip(), "source": path.name} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_encoder(self, model_name: str, cache_dir: str | None) -> None:
        if not model_name or model_name == "__offline__":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name, cache_folder=cache_dir)
        except (ImportError, OSError, RuntimeError):
            self.encoder = None

    def search(self, query: str, top_k: int = 8) -> list[Evidence]:
        if not self.documents:
            return []
        if self.encoder is None:
            tokens = set(query.lower().split())
            scored = [(sum(t in str(doc.get("text", "")).lower() for t in tokens), doc) for doc in self.documents]
        else:
            import numpy as np
            vectors = self.encoder.encode([query] + [str(doc.get("text", "")) for doc in self.documents], normalize_embeddings=True)
            scores = np.asarray(vectors[1:]) @ np.asarray(vectors[0])
            scored = list(zip(scores.tolist(), self.documents))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [Evidence(str(doc.get("source", "RAG")), str(doc.get("text", "")), float(score), "RAG", doc.get("metadata", {})) for score, doc in scored[:top_k]]
