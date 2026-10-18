"""Lightweight KG/RAG evidence stores used by the standalone agent."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str
    score: float = 0.0
    metadata: Mapping[str, Any] | None = None


class JsonlEvidenceStore:
    """Search a JSONL file without requiring a vector database.

    Each line must contain ``text`` and may contain ``source``, ``score`` and
    ``metadata``.  The same interface can later be replaced by a real KG or RAG
    backend without changing the agent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        if not self.path.exists():
            return []
        tokens = {token.lower() for token in query.split() if token.strip()}
        results: list[Evidence] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {self.path}") from exc
                text = str(item.get("text", ""))
                overlap = sum(token in text.lower() for token in tokens)
                if overlap or not tokens:
                    results.append(Evidence(
                        source=str(item.get("source", self.path.name)),
                        text=text,
                        score=float(item.get("score", overlap)),
                        metadata=item.get("metadata"),
                    ))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def __iter__(self) -> Iterable[Evidence]:
        return iter(self.search("", limit=10_000))


def evidence_to_dict(items: Iterable[Evidence]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
