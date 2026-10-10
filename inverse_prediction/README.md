# Inverse perturbation prediction

This isolated module predicts the perturbation gene from a post-perturbation expression change. It reuses the cached TxPert K562 split and combines a global expression view, salient-gene view, gene identity features, and STRING neighborhood propagation.

Run from the repository root:

```bash
python inverse_prediction/train_inverse.py --epochs 20 --out inverse_prediction/runs/k562_inverse_v2.pt
```

The script reports Recall@1/5/10 and MRR and writes a checkpoint plus a JSON metrics file. The first run requires the project dependencies and the cached K562 data.

The validation split contains perturbations not seen during training. Treat its Top-k score as an unseen-gene generalization measure, separate from training-set fit.
