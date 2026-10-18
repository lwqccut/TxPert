"""Forward model whose perturbation representation is augmented by LLM features."""
from __future__ import annotations

from typing import Iterable
import torch
from torch import nn

from forward_model.model import ConditionedMultiViewPerturbationModel


class KnowledgeAugmentedPerturbationModel(nn.Module):
    """Fuse frozen/precomputed knowledge vectors with the graph perturbation effect.

    The base model remains the existing multi-view forward model.  A learned
    projection and gated residual make the knowledge vector part of the loss,
    rather than an explanation attached after prediction.
    """
    def __init__(self, *, knowledge_embeddings: torch.Tensor, input_dim: int,
                 output_dim: int, num_nodes: int, edge_indices: list[torch.Tensor],
                 hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.1,
                 use_gene_attention: bool = True, use_multiview: bool = True,
                 use_hierarchical: bool = True, freeze_base: bool = False):
        super().__init__()
        if knowledge_embeddings.ndim != 2 or knowledge_embeddings.shape[0] < num_nodes:
            raise ValueError("knowledge_embeddings must have shape [num_nodes, knowledge_dim]")
        self.base = ConditionedMultiViewPerturbationModel(
            input_dim, output_dim, num_nodes, edge_indices, hidden_dim, latent_dim,
            dropout, use_gene_attention, use_multiview, use_hierarchical,
        )
        self.register_buffer("knowledge_embeddings", knowledge_embeddings[:num_nodes].float())
        knowledge_dim = int(knowledge_embeddings.shape[1])
        self.knowledge_projection = nn.Sequential(
            nn.LayerNorm(knowledge_dim), nn.Linear(knowledge_dim, latent_dim), nn.GELU()
        )
        self.control_gate = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, latent_dim), nn.Sigmoid()
        )
        self.knowledge_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _ids(items: Iterable, num_nodes: int, device: torch.device) -> list[list[int]]:
        rows = []
        for item in items:
            if torch.is_tensor(item):
                item = item.detach().cpu().flatten().tolist()
            elif not isinstance(item, (list, tuple)):
                item = [item]
            rows.append([int(value) for value in item if isinstance(value, (int, float)) and 0 <= int(value) < num_nodes])
        return rows

    def forward(self, control: torch.Tensor, pert_idxs: Iterable):
        base_prediction = self.base(control, pert_idxs)
        rows = self._ids(pert_idxs, self.knowledge_embeddings.shape[0], control.device)
        knowledge = []
        for ids in rows:
            if ids:
                knowledge.append(self.knowledge_embeddings[ids].mean(dim=0))
            else:
                knowledge.append(self.knowledge_embeddings.new_zeros(self.knowledge_embeddings.shape[1]))
        knowledge = torch.stack(knowledge).to(control.device)
        projected = self.knowledge_projection(knowledge)
        gate = self.control_gate(control[:, : self.base.output_dim])
        delta = self.knowledge_decoder(projected * gate)
        return base_prediction + delta

