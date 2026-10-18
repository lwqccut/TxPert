"""Forward-model backends used by the agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import copy


class TxPertCheckpointModel:
    """Run the released TxPert model on the repository's processed dataset.

    The datamodule, gene index, graph and checkpoint are all loaded from the
    original TxPert project.  No synthetic expression is created here.
    """
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from omegaconf import OmegaConf
        from gspp.data.datamodule import PertDataModule
        from gspp.data.graphmodule import GSPGraph
        from gspp.predictor import PertPredictor
        from gspp.constants import CHECKPOINT_DIR

        self.device = str(config.get("device", "auto"))
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        project_root = Path(__file__).resolve().parents[1]
        dm_cfg = copy.deepcopy(dict(config.get("datamodule", {})))
        dm_cfg.setdefault("mode", "inference")
        dm_cfg.setdefault("task_type", "K562_single_cell_line")
        dm_cfg.setdefault("match_cntr", True)
        dm_cfg.setdefault("avg_cntr", True)
        dm_cfg.setdefault("embed_cntr", True)
        dm_cfg.setdefault("obsm_key", "raw")
        self.datamodule = PertDataModule(**dm_cfg)
        self.datamodule.prepare_data()
        self.datamodule.setup("test")

        graph_cfg = copy.deepcopy(config.get("graph_cfg", {
            "graph1": {"graph_type": "string", "reduce2perts": True,
                        "norm_weights": False, "mode": "top_20"}
        }))
        graph = GSPGraph(self.datamodule.pert2id, self.datamodule.gene2id, graph_cfg)
        model_args = OmegaConf.create(copy.deepcopy(config.get("model", {})))
        model_args = OmegaConf.to_container(model_args, resolve=True)
        checkpoint = Path(str(config.get("checkpoint", "K562_unseen_pert_gat.ckpt")))
        if not checkpoint.is_absolute():
            checkpoint = project_root / "cache" / "checkpoints" / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"TxPert checkpoint not found: {checkpoint}")
        self.predictor = PertPredictor.load_from_checkpoint(
            checkpoint,
            input_dim=self.datamodule.input_dim,
            output_dim=self.datamodule.output_dim,
            adata_output_dim=self.datamodule.adata_output_dim,
            num_perts=len(self.datamodule.pert2id),
            graph=graph,
            pert_names=list(self.datamodule.pert2id.keys()),
            model_args=model_args,
            device=self.device,
        ).to(self.device).eval()
        self.control = self._mean_control()

    def _mean_control(self):
        import numpy as np
        matrix = self.datamodule.control_data_test.X
        return np.asarray(matrix.mean(axis=0)).reshape(-1).tolist()

    def resolve_perturbations(self, genes: Sequence[str]) -> list[int]:
        ids = []
        for gene in genes:
            candidates = [gene, f"{gene}+ctrl"]
            found = next((self.datamodule.pert2id[name] for name in candidates if name in self.datamodule.pert2id), None)
            if found is None:
                raise ValueError(f"Perturbation {gene!r} is not present in the TxPert gene set")
            ids.append(int(found))
        return ids

    def default_control(self) -> list[float]:
        return self.control

    def analysis_context(self, genes: Sequence[str]) -> dict[str, Any]:
        """Return a compact profile for Qwen, never the full expression matrix."""
        import numpy as np
        names = [str(item).upper() for item in self.datamodule.adata.var_names]
        values = np.asarray(self.control)
        selected = {
            gene: float(values[index]) for index, gene in enumerate(names) if gene in genes
        }
        top = np.argsort(values)[-20:][::-1]
        return {
            "cell_type": self.datamodule.test_cell_type,
            "n_genes": len(names),
            "selected_gene_control_values": selected,
            "top_control_genes": [names[index] for index in top],
            "knowledge_levels": {
                gene: int(self.datamodule.adata.uns.get("pharos", [-1] * len(names))[index])
                for index, gene in enumerate(names) if gene in genes
            },
        }

    def __call__(self, control: Sequence[float], perturbation_ids: Sequence[int], context: Mapping[str, Any]):
        import torch
        if len(control) != self.datamodule.input_dim:
            raise ValueError(f"control_expression has {len(control)} values; expected {self.datamodule.input_dim}")
        control_tensor = torch.as_tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_emb = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            prediction = self.predictor.sample_inference(
                control_tensor, [list(map(int, perturbation_ids))], p_emb,
                [self.datamodule.test_cell_type],
            )
        return prediction[0].detach().cpu()


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


class CustomForwardCheckpointModel:
    """Load the model trained by ``forward_model/train.py``.

    The artifact must contain ``model_spec`` (new training runs do). Older
    artifacts without graph metadata are rejected instead of silently producing
    predictions with a different graph.
    """
    def __init__(self, config: Mapping[str, Any]):
        import torch
        from forward_model.model import ConditionedMultiViewPerturbationModel

        checkpoint = Path(str(config["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Custom forward checkpoint not found: {checkpoint}")
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec = artifact.get("model_spec")
        if not spec:
            # Older forward_model/train.py artifacts only stored CLI arguments.
            # Reconstruct their graph and dimensions from the same TxPert cache.
            spec = self._infer_legacy_spec(artifact, config)
        if "edge_indices" not in spec:
            raise ValueError("Custom forward checkpoint is missing graph edge metadata")
        edges = [torch.as_tensor(edge, dtype=torch.long) for edge in spec["edge_indices"]]
        self.device = str(config.get("device", "auto"))
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ConditionedMultiViewPerturbationModel(
            input_dim=int(spec["input_dim"]), output_dim=int(spec["output_dim"]),
            num_nodes=int(spec["num_nodes"]), edge_indices=edges,
            hidden_dim=int(spec.get("hidden_dim", 256)), latent_dim=int(spec.get("latent_dim", 128)),
            dropout=float(spec.get("dropout", 0.1)),
            use_gene_attention=bool(spec.get("use_gene_attention", True)),
            use_multiview=bool(spec.get("use_multiview", True)),
            use_hierarchical=bool(spec.get("use_hierarchical", True)),
        )
        self.model.load_state_dict(artifact["model"], strict=True)
        self.model.to(self.device).eval()
        self.input_dim = int(spec["input_dim"])
        self.output_dim = int(spec["output_dim"])
        self.num_nodes = int(spec["num_nodes"])
        self._gene_map = {str(k).upper(): int(v) for k, v in dict(config.get("gene_to_id", {})).items()}
        self.control = list(map(float, config["control_expression"])) if config.get("control_expression") else None
        if self.control is None:
            from gspp.data.datamodule import PertDataModule
            import numpy as np
            dm_cfg = copy.deepcopy(dict(config.get("datamodule", {})))
            dm_cfg.setdefault("mode", "inference")
            dm_cfg.setdefault("task_type", "K562_single_cell_line")
            dm_cfg.setdefault("match_cntr", True)
            dm_cfg.setdefault("avg_cntr", True)
            dm_cfg.setdefault("embed_cntr", True)
            dm_cfg.setdefault("obsm_key", "raw")
            dm = PertDataModule(**dm_cfg)
            dm.prepare_data()
            self.control = np.asarray(dm.control_data_test.X.mean(axis=0)).reshape(-1).tolist()
            if not self._gene_map:
                self._gene_map = {str(k).upper(): int(v) for k, v in dm.pert2id.items()}

    @staticmethod
    def _infer_legacy_spec(artifact: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from gspp.data.datamodule import PertDataModule
        from gspp.data.graphmodule import GSPGraph
        state = artifact.get("model", artifact)
        args = artifact.get("config", {})
        views = args.get("views", config.get("views", ["string", "go"]))
        dm_cfg = copy.deepcopy(dict(config.get("datamodule", {})))
        dm_cfg.setdefault("mode", "train")
        dm_cfg.setdefault("task_type", "K562_single_cell_line")
        dm_cfg.setdefault("match_cntr", True)
        dm_cfg.setdefault("avg_cntr", True)
        dm_cfg.setdefault("embed_cntr", True)
        dm_cfg.setdefault("obsm_key", "raw")
        dm = PertDataModule(**dm_cfg)
        dm.prepare_data()
        graph = GSPGraph(dm.pert2id, dm.gene2id, {
            name: {"graph_type": name, "reduce2perts": True, "mode": "top_20"}
            for name in views
        })
        input_dim = int(state["cell_encoder.1.weight"].shape[1])
        output_dim = int(state["decoder.3.weight"].shape[0])
        latent_dim = int(state["gene_embeddings.0"].shape[1])
        hidden_dim = int(state["decoder.0.weight"].shape[0])
        return {
            "input_dim": input_dim, "output_dim": output_dim,
            "num_nodes": int(state["gene_embeddings.0"].shape[0]),
            "hidden_dim": hidden_dim, "latent_dim": latent_dim,
            "dropout": 0.1,
            "use_gene_attention": bool(args.get("use_gene_attention", True)),
            "use_multiview": bool(args.get("use_multiview", True)),
            "use_hierarchical": bool(args.get("use_hierarchical", True)),
            "views": views,
            "edge_indices": [torch.as_tensor(graph.graph_dict[name][0], dtype=torch.long).tolist() for name in views],
        }

    def default_control(self) -> list[float]:
        if self.control is None:
            raise ValueError("Custom forward model requires forward.control_expression")
        return self.control

    def resolve_perturbations(self, genes: Sequence[str]) -> list[int]:
        mapping = self._gene_map
        missing = [gene for gene in genes if gene.upper() not in mapping]
        if missing:
            raise ValueError("Custom forward model needs forward.gene_to_id entries for: " + ", ".join(missing))
        return [mapping[gene.upper()] for gene in genes]

    def __call__(self, control: Sequence[float], perturbation_ids: Sequence[int], context: Mapping[str, Any]):
        import torch
        if len(control) != self.input_dim:
            raise ValueError(f"control_expression has {len(control)} values; expected {self.input_dim}")
        if any(int(item) >= self.num_nodes for item in perturbation_ids):
            raise ValueError("perturbation id is outside the custom forward model graph")
        x = torch.as_tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            return self.model(x, [list(map(int, perturbation_ids))])[0].cpu()


def build_forward_model(config: Mapping[str, Any], *, input_dim: int | None = None,
                        output_dim: int | None = None, num_nodes: int | None = None):
    backend = str(config.get("backend", "txpert"))
    if backend == "txpert":
        return TxPertCheckpointModel(config)
    if backend == "demo":
        raise ValueError("demo backend is disabled for real experiments; configure txpert or custom_forward")
    if backend == "custom_forward":
        model = CustomForwardCheckpointModel(config)
        return model
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
