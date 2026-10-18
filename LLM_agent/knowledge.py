"""Local KG loader for STRING/GO/Reactome-style edge tables."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv

from .schemas import Evidence


class KnowledgeGraph:
    def __init__(self, path: str | Path | None = None):
        self.edges: list[tuple[str, str, float, str]] = []
        if path and Path(path).exists():
            self._load(Path(path))

    def _load(self, path: Path) -> None:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = row.get("source") or row.get("gene1") or row.get("regulator")
                target = row.get("target") or row.get("gene2")
                if source and target:
                    self.edges.append((source.upper(), target.upper(), float(row.get("weight", 1.0)), row.get("database", "KG")))

    def query(self, genes: Iterable[str], top_k: int = 8) -> list[Evidence]:
        query = {str(item).upper() for item in genes}
        hits = []
        for source, target, weight, database in self.edges:
            if source in query or target in query:
                hits.append(Evidence(database, f"{source} is connected to {target}.", weight, "KG",
                                      {"source": source, "target": target}))
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
