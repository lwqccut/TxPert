# Forward multi-view ablations

This directory contains isolated forward-prediction experiments. The original `main.py`, `configs/`, and `inverse_prediction/` workflow remain separate.

Run each experiment from the repository root. All experiments use the same K562 split and differ only in the stated component:

```bash
python forward_ablation/01_baseline_exphormer.py --epochs 1
python forward_ablation/02_multiview_string_go.py --epochs 1
```

`01` uses one STRING graph. `02` uses STRING and GO as two graph views with the existing multi-graph Exphormer implementation. The scripts skip the Zenodo request when the local cache is present and write outputs under `forward_ablation/runs/`.
