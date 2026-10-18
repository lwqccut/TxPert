"""Train the knowledge-augmented model on the original TxPert dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from gspp.data.datamodule import PertDataModule
from gspp.data.graphmodule import GSPGraph
from .knowledge_model import KnowledgeAugmentedPerturbationModel


def graph_edges(dm, views):
    graph = GSPGraph(dm.pert2id, dm.gene2id, {
        name: {"graph_type": name, "reduce2perts": True, "mode": "top_20"}
        for name in views
    })
    return [torch.as_tensor(graph.graph_dict[name][0], dtype=torch.long) for name in views]


def evaluate(model, loader, device, output_dim):
    model.eval(); losses = []; correlations = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch.control.to(device), batch.pert_idxs)
            target = batch.x[:, :output_dim].to(device)
            losses.append(torch.mean((prediction - target) ** 2).item())
            for pred, truth in zip(prediction, target):
                correlations.append(torch.corrcoef(torch.stack([pred, truth]))[0, 1].item())
    return {"mse": sum(losses) / max(1, len(losses)), "pearson": sum(correlations) / max(1, len(correlations))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("LLM_agent/runs/knowledge_features.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--views", nargs="+", default=["string", "go"])
    parser.add_argument("--out", type=Path, default=Path("LLM_agent/runs/knowledge_forward.pt"))
    args = parser.parse_args()
    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = PertDataModule(batch_size=args.batch_size, match_cntr=True, avg_cntr=True, embed_cntr=True,
                        obsm_key="raw", task_type="K562_single_cell_line", mode="train")
    dm.prepare_data(); dm.setup("fit")
    edges = graph_edges(dm, args.views)
    model = KnowledgeAugmentedPerturbationModel(
        knowledge_embeddings=artifact["embeddings"], input_dim=dm.input_dim,
        output_dim=dm.adata_output_dim, num_nodes=len(dm.pert2id), edge_indices=edges,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for step, batch in enumerate(dm.train_dataloader()):
            prediction = model(batch.control.to(device), batch.pert_idxs)
            target = batch.x[:, :dm.adata_output_dim].to(device)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item()
        metrics = evaluate(model, dm.val_dataloader(), device, dm.adata_output_dim)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total / max(1, step + 1), **metrics}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "features": str(args.features),
                "model_spec": {"input_dim": dm.input_dim, "output_dim": dm.adata_output_dim,
                               "num_nodes": len(dm.pert2id), "views": args.views}}, args.out)


if __name__ == "__main__":
    main()
