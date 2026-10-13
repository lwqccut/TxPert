"""Training entrypoint for the configurable forward model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from gspp.data.datamodule import PertDataModule
from gspp.data.graphmodule import GSPGraph
from .model import ConditionedMultiViewPerturbationModel


def graph_edges(dm, views):
    graph = GSPGraph(dm.pert2id, dm.gene2id, {
        name: {"graph_type": name, "reduce2perts": True, "mode": "top_20"}
        for name in views
    })
    return [torch.as_tensor(graph.graph_dict[name][0], dtype=torch.long) for name in views]


def evaluate(model, loader, device, output_dim):
    model.eval(); se = []; corrs = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.control.to(device), batch.pert_idxs)
            target = batch.x[:, :output_dim].to(device)
            se.append(torch.mean((pred - target) ** 2).item())
            for p, y in zip(pred, target):
                corrs.append(torch.corrcoef(torch.stack([p, y]))[0, 1].item())
    return {"mse": sum(se) / max(1, len(se)), "pearson": sum(corrs) / max(1, len(corrs))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--use-gene-attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-multiview", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-hierarchical", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-view-consistency", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--views", nargs="+", default=["string", "go"])
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--out", default="forward_model/runs/model.pt")
    args = p.parse_args(); torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = PertDataModule(batch_size=args.batch_size, match_cntr=True, avg_cntr=True, embed_cntr=True, obsm_key="raw", task_type="K562_single_cell_line", mode="train")
    dm.prepare_data(); dm.setup("fit")
    edges = graph_edges(dm, args.views)
    model = ConditionedMultiViewPerturbationModel(dm.input_dim, dm.adata_output_dim, len(dm.pert2id), edges, use_gene_attention=args.use_gene_attention, use_multiview=args.use_multiview, use_hierarchical=args.use_hierarchical).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = dm.train_dataloader()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for step, batch in enumerate(loader):
            pred = model(batch.control.to(device), batch.pert_idxs)
            target = batch.x[:, :dm.adata_output_dim].to(device)
            loss = torch.mean((pred - target) ** 2)
            if args.use_view_consistency: loss = loss + 0.01 * model.consistency_loss()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += loss.item()
            if args.max_steps and step + 1 >= args.max_steps: break
        val = {"mse": None, "pearson": None} if args.skip_eval else evaluate(model, dm.val_dataloader(), device, dm.adata_output_dim)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total / (step + 1), **val}), flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args), "val": val}, out)


if __name__ == "__main__": main()
