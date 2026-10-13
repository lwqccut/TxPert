"""Configurable forward perturbation model.

The model is intentionally independent from the original TxPert implementation.
It receives a control expression vector and perturbation ids and predicts the
perturbed expression vector. Each proposed innovation can be disabled for an
ablation run.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import torch
from torch import nn


class ConditionedMultiViewPerturbationModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        edge_indices: Sequence[torch.Tensor],
        hidden_dim: int = 256,
        latent_dim: int = 128,
        dropout: float = 0.1,
        use_gene_attention: bool = True,
        use_multiview: bool = True,
        use_hierarchical: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.use_gene_attention = use_gene_attention
        self.use_multiview = use_multiview
        self.use_hierarchical = use_hierarchical
        self.num_views = len(edge_indices)

        self.cell_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.gene_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(num_nodes, latent_dim) * 0.02) for _ in edge_indices]
        )
        self.register_buffer("edge_src", torch.cat([e[0] for e in edge_indices]) if edge_indices else torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_dst", torch.cat([e[1] for e in edge_indices]) if edge_indices else torch.empty(0, dtype=torch.long))
        offsets = []
        start = 0
        for e in edge_indices:
            offsets.append((start, start + e.shape[1]))
            start += e.shape[1]
        self._view_offsets = offsets

        self.view_attention = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1)
        )
        self.gene_attention = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1)
        )
        self.module_pool = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def _view_node_embeddings(self, view: int) -> torch.Tensor:
        """One message-passing step gives each gene a local graph representation."""
        h = self.gene_embeddings[view]
        if not self.edge_indices[view].numel():
            return h
        src, dst = self.edge_indices[view]
        agg = torch.zeros_like(h).index_add(0, dst, h[src])
        deg = torch.zeros(self.num_nodes, device=h.device).index_add(
            0, dst, torch.ones(dst.numel(), device=h.device)
        )
        return h + agg / deg.clamp_min(1).unsqueeze(1)

    @property
    def edge_indices(self):
        return [torch.stack((self.edge_src[a:b], self.edge_dst[a:b])) for a, b in self._view_offsets]

    def _parse_perturbations(self, pert_idxs: Iterable, device) -> torch.Tensor:
        rows = []
        for item in pert_idxs:
            if torch.is_tensor(item):
                values = item.detach().cpu().flatten().tolist()
            elif isinstance(item, (list, tuple)):
                values = list(item)
            else:
                values = [item]
            rows.append([int(v) for v in values if isinstance(v, (int, float)) and int(v) >= 0 and int(v) < self.num_nodes])
        return rows

    def forward(self, control: torch.Tensor, pert_idxs: Iterable):
        cell = self.cell_encoder(control[:, : self.output_dim])
        view_nodes = [self._view_node_embeddings(v) for v in range(self.num_views)]
        rows = self._parse_perturbations(pert_idxs, control.device)
        effects = []
        for i, ids in enumerate(rows):
            if not ids:
                effects.append(torch.zeros(cell.shape[1], device=control.device)); continue
            node_views = torch.stack([view_nodes[v][ids].mean(0) for v in range(self.num_views)])
            if self.use_multiview and self.num_views > 1:
                q = cell[i].expand(self.num_views, -1)
                weights = torch.softmax(self.view_attention(torch.cat([q, node_views], -1)).flatten(), 0)
                effect = (weights[:, None] * node_views).sum(0)
            else:
                effect = node_views[0]
            if self.use_gene_attention:
                gene_nodes = view_nodes[0][ids]
                q = cell[i].expand(gene_nodes.shape[0], -1)
                scores = torch.softmax(self.gene_attention(torch.cat([q, gene_nodes], -1)).flatten(), 0)
                effect = 0.5 * effect + 0.5 * (scores[:, None] * gene_nodes).sum(0)
            if self.use_hierarchical:
                effect = effect + self.module_pool(effect)
            effects.append(effect)
        effect = torch.stack(effects)
        return self.decoder(torch.cat([cell, effect], -1))

    def consistency_loss(self) -> torch.Tensor:
        """Keeps view embeddings comparable while allowing view-specific features."""
        if self.num_views < 2:
            return self.gene_embeddings[0].new_zeros(())
        a = self.gene_embeddings[0] / self.gene_embeddings[0].norm(dim=1, keepdim=True).clamp_min(1e-6)
        b = self.gene_embeddings[1] / self.gene_embeddings[1].norm(dim=1, keepdim=True).clamp_min(1e-6)
        return (1 - (a * b).sum(1)).mean()
