"""Adapter for an already-loaded PyTorch forward model.

Loading checkpoints remains the responsibility of the caller because TxPert has
several model configurations.  This adapter only standardizes the call shape
expected by :class:`LLMAgent`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class TxPertForwardAdapter:
    def __init__(self, model: Any, device: str = "cpu"):
        import torch
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def __call__(self, control_expression: Sequence[float], perturbation_ids: Sequence[int], context: Mapping[str, Any]):
        import torch
        control = torch.as_tensor(control_expression, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            # ConditionedMultiViewPerturbationModel accepts one perturbation-id
            # list per sample.  Keep the adapter compatible with that contract.
            output = self.model(control, [list(perturbation_ids)])
        return output.squeeze(0).detach().cpu()
