"""BGE-M3 RAG index with a lexical fallback for CPU/offline smoke tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .schemas import Evidence


class RAGIndex:
    def __init__(self, paths: str | Path | Sequence[str | Path] | None = None, embedding_model: str = "BAAI/bge-m3", cache_dir: str | None = None):
        self.documents: list[dict[str, Any]] = []
        self.encoder = None
        if paths is None:
            paths = []
        elif isinstance(paths, (str, Path)):
            paths = [paths]
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"RAG corpus file not found: {path}")
            self._load(path)
        if self.documents:
            self._load_encoder(embedding_model, cache_dir)

    def _load(self, path: Path) -> None:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                self.documents = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix.lower() in {".md", ".txt", ".rst"}:
            words = path.read_text(encoding="utf-8", errors="ignore").split()
            for start in range(0, len(words), 350):
                text = " ".join(words[start:start + 350])
                if text:
                    self.documents.append({"text": text, "source": str(path), "metadata": {"offset": start}})
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("pypdf is required to index PDF literature") from exc
            reader = PdfReader(str(path))
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    self.documents.append({"text": text, "source": str(path), "metadata": {"page": number}})
        else:
            raise ValueError(f"Unsupported RAG corpus format: {path.suffix}")

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
