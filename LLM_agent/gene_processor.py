"""Gene symbol normalization and task extraction from user language."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .schemas import GeneTask


class GeneProcessor:
    def __init__(self, aliases: dict[str, str] | None = None, gene_to_id: dict[str, int] | None = None):
        self.aliases = {key.upper(): value.upper() for key, value in (aliases or {}).items()}
        self.gene_to_id = {key.upper(): int(value) for key, value in (gene_to_id or {}).items()}

    def normalize(self, genes: Iterable[str]) -> list[str]:
        normalized = []
        for gene in genes:
            symbol = re.sub(r"[^A-Za-z0-9-]", "", str(gene)).upper()
            symbol = self.aliases.get(symbol, symbol)
            if symbol and symbol not in normalized:
                normalized.append(symbol)
        return normalized

    def resolve_ids(self, genes: Iterable[str]) -> list[int]:
        return [self.gene_to_id[g] for g in self.normalize(genes) if g in self.gene_to_id]

    def extract(self, prompt: str, *, cell_type: str = "K562") -> GeneTask:
        # Gene symbols are intentionally conservative: all-caps symbols with 2+
        # characters, avoiding ordinary prose words through a small stoplist.
        stop = {"THE", "AND", "FOR", "WITH", "GENE", "CELL", "K562"}
        candidates = re.findall(r"\b[A-Z][A-Z0-9-]{1,14}\b", prompt.upper())
        genes = self.normalize(item for item in candidates if item not in stop)
        return GeneTask(prompt=prompt, cell_type=cell_type, genes=genes, perturbation_ids=self.resolve_ids(genes))

    @classmethod
    def from_tsv(cls, path: str | Path) -> "GeneProcessor":
        aliases, mapping = {}, {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    mapping[fields[0]] = int(fields[1])
                if len(fields) >= 3:
                    aliases[fields[2]] = fields[0]
        return cls(aliases=aliases, gene_to_id=mapping)
