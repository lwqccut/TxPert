"""Forward model with key-gene LLM features as a special input branch."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from forward_model.model import ConditionedMultiViewPerturbationModel


class KeyGeneAugmentedPerturbationModel(nn.Module):
    """Fuse frozen key-gene embeddings with the existing multi-view model.

    The original STRING/GO graph views remain unchanged.  The selected-gene
    LLM feature is a perturbation-level special input and contributes a gated
    residual to the predicted expression.  Its projection and decoder are
    trained; the precomputed embedding itself is frozen.
    """

    def __init__(self, *, key_embeddings: torch.Tensor, key_mask: torch.Tensor | None,
                 input_dim: int, output_dim: int, num_nodes: int,
                 edge_indices: list[torch.Tensor], hidden_dim: int = 256,
                 latent_dim: int = 128, dropout: float = 0.1,
                 use_gene_attention: bool = True, use_multiview: bool = True,
                 use_hierarchical: bool = True, freeze_base: bool = False):
        super().__init__()
        if key_embeddings.ndim != 2 or key_embeddings.shape[0] < num_nodes:
            raise ValueError("key_embeddings must have shape [num_nodes, feature_dim]")
        if key_mask is not None and key_mask.shape[0] < num_nodes:
            raise ValueError("key_mask must contain one entry per perturbation node")
        self.base = ConditionedMultiViewPerturbationModel(
            input_dim, output_dim, num_nodes, edge_indices, hidden_dim, latent_dim,
            dropout, use_gene_attention, use_multiview, use_hierarchical,
        )
        self.register_buffer("key_embeddings", key_embeddings[:num_nodes].float())
        mask = torch.ones(num_nodes, dtype=torch.float32) if key_mask is None else key_mask[:num_nodes].float()
        self.register_buffer("key_mask", mask)
        feature_dim = int(key_embeddings.shape[1])
        self.key_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, latent_dim), nn.GELU()
        )
        self.control_gate = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, latent_dim), nn.Sigmoid()
        )
        self.key_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _ids(items: Iterable, num_nodes: int) -> list[list[int]]:
        rows = []
        for item in items:
            if torch.is_tensor(item):
                item = item.detach().cpu().flatten().tolist()
            elif not isinstance(item, (list, tuple)):
                item = [item]
            rows.append([
                int(value) for value in item
                if isinstance(value, (int, float)) and 0 <= int(value) < num_nodes
            ])
        return rows

    def forward(self, control: torch.Tensor, pert_idxs: Iterable):
        base_prediction = self.base(control, pert_idxs)
        rows = self._ids(pert_idxs, self.key_embeddings.shape[0])
        features = []
        available = []
        for ids in rows:
            if ids:
                features.append(self.key_embeddings[ids].mean(dim=0))
                available.append(self.key_mask[ids].mean().clamp(0.0, 1.0))
            else:
                features.append(self.key_embeddings.new_zeros(self.key_embeddings.shape[1]))
                available.append(self.key_embeddings.new_zeros(()))
        key_features = torch.stack(features).to(control.device)
        availability = torch.stack(available).to(control.device).unsqueeze(1)
        projected = self.key_projection(key_features) * availability
        gate = self.control_gate(control[:, : self.base.output_dim])
        delta = self.key_decoder(projected * gate) * availability
        return base_prediction + delta

