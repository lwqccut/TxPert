"""Reuse the original architecture, training loss and scheduler configuration."""
import torch
from gspp.predictor import PertPredictor
from .core import group_parameters


class ReactivePredictor(PertPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The original TxPert constructor converts numpy edges with torch.Tensor,
        # yielding float32 indices; PyG requires integers. Normalize only graph
        # index dtype, preserving the graph, model weights and architecture.
        edges = getattr(self.model, 'edge_index', None)
        if torch.is_tensor(edges):
            self.model.edge_index = edges.long()
        elif isinstance(edges, dict):
            self.model.edge_index = {key: value.long() for key, value in edges.items()}
        if hasattr(self.model, 'graphs'):
            self.model.graphs = [(edge.long(), weight, nodes)
                                 for edge, weight, nodes in self.model.graphs]
        if hasattr(self.model, 'g_supra'):
            edge, weight, nodes = self.model.g_supra
            self.model.g_supra = (edge.long(), weight, nodes)

    def configure_optimizers(self):
        # Preserve the original optimizer/scheduler recipe while expanding its groups.
        original = super().configure_optimizers()
        optimizer = original['optimizer'] if isinstance(original, dict) else original
        template = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        optimizer.param_groups.clear()
        for group in group_parameters(self.model):
            optimizer.add_param_group({**template, **group})
        if isinstance(original, dict):
            scheduler = original['lr_scheduler']
            # TxPert schedules are created with one group; propagate their initial
            # per-group arrays, including children of SequentialLR.
            def expand(item):
                for attr in ('base_lrs', '_last_lr'):
                    if hasattr(item, attr):
                        values = getattr(item, attr)
                        setattr(item, attr, values * len(optimizer.param_groups))
                for child in getattr(item, '_schedulers', []):
                    expand(child)
            expand(scheduler)
        return original

    def validation_step(self, batch, batch_idx):
        prediction, _, _ = self.forward(batch.control, batch.pert_idxs, batch.p)
        target = batch.x[:, :self.adata_output_dim].to(prediction.device)
        self.log('reactive_val_mse', torch.mean((prediction - target) ** 2),
                 on_step=False, on_epoch=True, batch_size=prediction.shape[0])

    def on_validation_epoch_end(self):
        # Parent evaluates test_dataloader during validation. Deliberately omit it:
        # the controller and checkpoint selection must never observe test results.
        pass
