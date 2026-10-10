"""Inverse perturbation prediction (expression change -> causal gene).

This module is intentionally isolated from the original TxPert entrypoints.
It reuses the project's cached K562 splits and supports top-k gene retrieval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gspp.data.datamodule import PertDataModule
from gspp.data.graphmodule import GSPGraph


class MultiViewInverseModel(nn.Module):
    """Expression, salient-gene and candidate-gene views with graph-free scoring."""
    def __init__(self, n_genes: int, n_candidates: int, edge_index=None, candidate_indices=None, hidden: int = 256, emb: int = 128):
        super().__init__()
        self.n_candidates = n_candidates
        self.expr = nn.Sequential(nn.LayerNorm(n_genes), nn.Linear(n_genes, hidden),
                                  nn.GELU(), nn.Dropout(.15), nn.Linear(hidden, emb))
        self.salient = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, emb))
        self.candidate = nn.Embedding(n_candidates, emb)
        self.temperature = nn.Parameter(torch.tensor(.07))
        self.gene_identity = nn.Embedding(n_genes + 1, emb)
        self.register_buffer('candidate_indices', candidate_indices)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('edge_index', edge_index if edge_index is not None else torch.empty((2, 0), dtype=torch.long))

    def forward(self, delta: torch.Tensor, salient_idx: torch.Tensor):
        z = self.expr(delta)
        # A second view retains the strongest expression changes explicitly.
        vals = torch.gather(delta, 1, salient_idx)
        identity_view = (self.gene_identity(salient_idx) * vals.unsqueeze(-1)).mean(1)
        z = z + self.salient(vals) + identity_view
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        c = self.gene_identity(self.candidate_indices)
        if self.edge_index.numel():
            src, dst = self.edge_index
            agg = torch.zeros_like(c).index_add(0, dst, c[src])
            deg = torch.zeros(c.shape[0], device=c.device).index_add(0, dst, torch.ones_like(dst, dtype=torch.float))
            c = c + 0.25 * agg / deg.clamp_min(1).unsqueeze(1)
        c = c / (c.norm(dim=-1, keepdim=True) + 1e-6)
        scores = z @ c.T / self.temperature.clamp(.02, .5)
        # CRISPR knockdown prior; unavailable expression genes receive zero prior.
        valid = self.candidate_indices < delta.shape[1]
        prior = -delta[:, self.candidate_indices.clamp_max(delta.shape[1]-1)] * valid
        return scores + self.prior_scale * prior


def collect(dm, split: str):
    ds = getattr(dm, f"{split}_data")
    # Dataset tensors are already cached and avoid loading individual cells repeatedly.
    labels = torch.tensor([int(x[0]) if hasattr(x, '__len__') else int(x) for x in ds.pert_conditions])
    keep = labels >= 0
    # Use condition-level means: inverse prediction targets the perturbation,
    # so this removes cell noise and reduces 100k cells to ~1k examples.
    raw = ds.pert_data[keep].float()
    yy = labels[keep]
    ctrl = ds.control_data.mean(0)
    raw = raw - ctrl
    unique = torch.unique(yy, sorted=True)
    sums = torch.zeros(len(unique), raw.shape[1])
    group = torch.searchsorted(unique, yy)
    sums.index_add_(0, group, raw)
    means = sums / torch.bincount(group).unsqueeze(1)
    return means, unique


def metrics(logits, y):
    order = logits.argsort(dim=1, descending=True)
    out = {}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = (order[:, :k] == y[:, None]).any(1).float().mean().item()
    out["mrr"] = (1.0 / ((order == y[:, None]).nonzero()[:, 1].float() + 1)).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-batches', type=int, default=0)
    ap.add_argument('--out', default='inverse_prediction/runs/k562_inverse_v2.pt')
    args = ap.parse_args()
    torch.manual_seed(1)
    torch.set_num_threads(2)
    dm = PertDataModule(batch_size=args.batch_size, match_cntr=True, avg_cntr=True,
                        embed_cntr=True, obsm_key='raw', task_type='K562_single_cell_line', mode='train')
    dm.prepare_data(); dm.setup('fit')
    xtr, ytr = collect(dm, 'train'); xva, yva = collect(dm, 'val')
    n_genes = xtr.shape[1]; n_candidates = len(dm.pert2id) - 1
    # p indices are dataset ids; remap sparse ids to [0, n_candidates).
    expression_lookup = {str(name): i for i, name in enumerate(dm.adata.var_names)}
    # Direct Top-5 mode: retain only genes with an expression feature. Genes
    # absent from the measured feature space cannot be identified from this input.
    candidate_orig = sorted(v for v in dm.id2pert if v >= 0 and dm.id2pert[v] in expression_lookup)
    remap = {v: i for i, v in enumerate(candidate_orig)}
    graph = GSPGraph(dm.pert2id, dm.gene2id, {'string': {'graph_type': 'string', 'reduce2perts': True, 'mode': 'top_20'}})
    raw_edges = graph.graph_dict['string'][0].long()
    id_to_class = {v: k for k, v in remap.items()}
    edges = torch.tensor([[remap[int(a)], remap[int(b)]] for a, b in raw_edges.T.tolist()
                          if int(a) in remap and int(b) in remap], dtype=torch.long).T if raw_edges.numel() else torch.empty((2, 0), dtype=torch.long)
    candidate_indices = torch.tensor([expression_lookup.get(dm.id2pert[v], n_genes) for v in remap])
    model = MultiViewInverseModel(n_genes, len(remap), edge_index=edges, candidate_indices=candidate_indices)
    print(json.dumps({'candidate_mode': 'expression_covered_direct_top5', 'candidates': len(remap), 'train_conditions': len(ytr), 'unseen_val_conditions': len(yva)}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Drop labels outside the measurable candidate space.
    valid_train = torch.tensor([int(v) in remap for v in ytr])
    valid_val = torch.tensor([int(v) in remap for v in yva])
    xtr, ytr = xtr[valid_train], torch.tensor([remap[int(v)] for v in ytr[valid_train]])
    xva, yva = xva[valid_val], torch.tensor([remap[int(v)] for v in yva[valid_val]])
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    salient_k = min(128, n_genes)
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for step, (x, y) in enumerate(loader):
            idx = x.abs().topk(salient_k, dim=1).indices
            loss = nn.functional.cross_entropy(model(x, idx), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
            if args.max_batches and step + 1 >= args.max_batches: break
        model.eval()
        with torch.no_grad():
            idx = xva.abs().topk(salient_k, dim=1).indices
            val_logits = model(xva, idx)
            result = metrics(val_logits, yva)
            result['val_loss'] = nn.functional.cross_entropy(val_logits, yva).item()
            result['train_recall@1'] = metrics(model(xtr, xtr.abs().topk(salient_k, dim=1).indices), ytr)['recall@1']
        print(json.dumps({'epoch': epoch + 1, 'train_loss': total / (step + 1), **result}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'gene_to_class': remap, 'metrics': result}, args.out)
    Path(args.out).with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
