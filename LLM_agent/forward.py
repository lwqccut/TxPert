"""Forward-model backends used by the agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


class DemoForwardModel:
    """Deterministic fallback so the complete pipeline runs without a checkpoint."""
    def __call__(self, control: Sequence[float], perturbation_ids: Sequence[int], context: Mapping[str, Any]) -> list[float]:
        shift = 0.01 * sum(perturbation_ids)
        return [float(value) + shift for value in control]


class TorchForwardModel:
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, control: Sequence[float], perturbation_ids: Sequence[int], context: Mapping[str, Any]):
        import torch
        x = torch.as_tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(x, [list(perturbation_ids)])
        return output[0].detach().cpu()


def build_forward_model(config: Mapping[str, Any], *, input_dim: int | None = None,
                        output_dim: int | None = None, num_nodes: int | None = None):
    backend = str(config.get("backend", "demo"))
    if backend == "demo":
        return DemoForwardModel()
    if backend != "checkpoint":
        raise ValueError(f"Unsupported forward backend: {backend}")
    if not all(value is not None for value in (input_dim, output_dim, num_nodes)):
        raise ValueError("checkpoint backend requires input_dim, output_dim and num_nodes")
    import torch
    from forward_model.model import ConditionedMultiViewPerturbationModel
    checkpoint = Path(str(config["checkpoint"]))
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = artifact.get("model", artifact)
    # A checkpoint saved by forward_model/train.py contains graph edges in the
    # model state only when they were supplied.  Use an empty graph here; callers
    # doing a full paper run should pass their graph-built model directly.
    model = ConditionedMultiViewPerturbationModel(input_dim, output_dim, num_nodes, [])
    model.load_state_dict(state, strict=False)
    return TorchForwardModel(model, str(config.get("device", "cpu")))
