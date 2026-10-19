# LLM_agent_plus

This package adds a bounded, key-gene LLM feature branch to TxPert. The
original `LLM_agent` package, data paths, model cache, STRING/GO graph files,
Qwen client and BGE-M3 cache are reused. Only generated artifacts are written
under `LLM_agent_plus/runs`.

The LLM is not called for every gene in the expression matrix. The precompute
step ranks perturbation genes from the existing TxPert data using observed
response signal, the existing Pharos prior and perturbation frequency. It calls
Qwen only for the top `max_key_genes` genes, summarizes their existing STRING/GO
evidence, and embeds the summaries with the existing BGE-M3 model. Unselected
perturbations receive a zero vector and are marked in `key_mask`.

The training model keeps the original STRING and GO views. The selected-gene
embedding is a frozen perturbation-level special input, projected and gated by
the control expression, then decoded as a residual correction to the original
forward prediction. This keeps the LLM information separate and makes it easy
to compare against the original model.

From the repository root:

```bash
python -m LLM_agent_plus.precompute_key_features \
  --config LLM_agent_plus/config.yaml \
  --output LLM_agent_plus/runs/key_gene_features.pt

python -m LLM_agent_plus.train \
  --features LLM_agent_plus/runs/key_gene_features.pt \
  --out LLM_agent_plus/runs/key_gene_forward.pt
```

To reduce or increase the number of LLM calls, change
`key_gene_selection.max_key_genes` in `LLM_agent_plus/config.yaml`. The
precompute artifact records the selected genes and their data-derived scores.

