"""Query the repository's real STRING/GO/Reactome-style edge tables."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

from .schemas import Evidence


class KnowledgeGraph:
    def __init__(self, paths: str | Path | Sequence[str | Path] | None = None):
        if paths is None:
            self.paths = []
        elif isinstance(paths, (str, Path)):
            self.paths = [Path(paths)]
        else:
            self.paths = [Path(path) for path in paths]
        missing = [str(path) for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Knowledge graph files not found: {missing}")

    def _load(self, path: Path) -> None:
        raise RuntimeError("KnowledgeGraph loads files lazily; call query()")

    def query(self, genes: Iterable[str], top_k: int = 8) -> list[Evidence]:
        query = {str(item).upper() for item in genes}
        if not query:
            return []
        hits: list[Evidence] = []
        for path in self.paths:
            if path.suffix.lower() == ".parquet":
                # Predicate pushdown avoids materializing all 11M STRING edges.
                filters = [[("regulator", "in", list(query))], [("target", "in", list(query))]]
                frame = pd.read_parquet(path, columns=["regulator", "target", "weight"], filters=filters)
                database = path.stem
            else:
                chunks = []
                for frame in pd.read_csv(path, usecols=lambda c: c in {"source", "target", "regulator", "gene1", "gene2", "importance", "weight"}, chunksize=200_000):
                    source_col = "source" if "source" in frame else "regulator" if "regulator" in frame else "gene1"
                    target_col = "target" if "target" in frame else "gene2"
                    mask = frame[source_col].astype(str).str.upper().isin(query) | frame[target_col].astype(str).str.upper().isin(query)
                    if mask.any():
                        chunks.append(frame.loc[mask].copy())
                frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                database = path.stem
            if frame.empty:
                continue
            source_col = "regulator" if "regulator" in frame else "source"
            target_col = "target"
            weight_col = "weight" if "weight" in frame else "importance"
            for row in frame.itertuples(index=False):
                values = row._asdict()
                source, target = str(values[source_col]).upper(), str(values[target_col]).upper()
                hits.append(Evidence(database, f"{source} is connected to {target} in {database}.",
                                      float(values.get(weight_col, 1.0)), "KG",
                                      {"source": source, "target": target, "database": database}))
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
