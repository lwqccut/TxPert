"""Streaming validation Pearson delta, using control means and no test targets."""
import numpy as np
import torch

from gspp import constants as cs


def control_means(adata, match_batch=True):
    """Read only control rows; avoid the original metrics' global/test-target caches."""
    controls = adata[adata.obs[cs.CONDITION].astype(str) == cs.CONTROL_LABEL]
    groups = {}
    for index, (cell, batch) in enumerate(zip(controls.obs[cs.CELL_TYPE], controls.obs[cs.BATCH])):
        key = (str(cell), str(batch) if match_batch else None)
        groups.setdefault(key, []).append(index)
    means = {}
    for key, indices in groups.items():
        mean = np.asarray(controls.X[indices].mean(axis=0)).reshape(-1)
        means[key] = torch.as_tensor(mean, dtype=torch.float64)
    if not means:
        raise ValueError('Validation Pearson delta requires expression controls')
    return means


class ValidationDelta:
    """Equal mean over perturbation/dose/cell-type correlations, as in gspp.metrics."""
    def __init__(self, means, match_batch=True):
        self.means, self.match_batch = means, match_batch
        self.sums = {}

    def update(self, prediction, target, batch):
        pred, truth = prediction.detach().double().cpu(), target.detach().double().cpu()
        metadata = zip(batch.pert_cond_names, batch.pert_dose, batch.cell_types,
                       batch.experimental_batches)
        # Aggregate on the CPU without retaining individual cells or GPU tensors.
        grouped = {}
        for index, (condition, dose, cell, experimental_batch) in enumerate(metadata):
            key = (str(condition), str(dose), str(cell))
            control_key = (str(cell), str(experimental_batch) if self.match_batch else None)
            grouped.setdefault((key, control_key), []).append(index)
        for (key, control_key), indices in grouped.items():
            if control_key not in self.means:
                raise ValueError(f'Missing validation control mean for {control_key}')
            control = self.means[control_key]
            if control.numel() != pred.shape[1]:
                raise ValueError('Control expression and prediction gene dimensions differ')
            n = len(indices)
            sums = self.sums.setdefault(key, [torch.zeros_like(control), torch.zeros_like(control)])
            sums[0] += pred[indices].sum(0) - n * control
            sums[1] += truth[indices].sum(0) - n * control

    def compute(self):
        correlations, undefined = [], 0
        for predicted, observed in self.sums.values():
            if not torch.isfinite(predicted).all() or not torch.isfinite(observed).all():
                return float('nan'), len(self.sums), undefined
            x, y = predicted - predicted.mean(), observed - observed.mean()
            denominator = x.norm() * y.norm()
            if denominator == 0:
                correlations.append(0.0)  # Match original constant-vector convention.
                undefined += 1
            else:
                correlations.append(float((x @ y / denominator).clamp(-1, 1)))
        return (float(np.mean(correlations)) if correlations else float('nan'),
                len(correlations), undefined)
