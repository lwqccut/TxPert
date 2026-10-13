# Configurable forward model

The model and training entrypoint are separate. Each proposed innovation is a switch so that runs can be compared with one change at a time:

```bash
# baseline gene encoder, STRING only
python -m forward_model.train --views string --no-use-gene-attention --no-use-multiview --no-use-hierarchical

# gene attention only
python -m forward_model.train --views string --use-gene-attention --no-use-multiview --no-use-hierarchical

# multi-view STRING + GO
python -m forward_model.train --views string go --use-gene-attention --use-multiview --no-use-hierarchical

# full model
python -m forward_model.train --views string go --use-gene-attention --use-multiview --use-hierarchical --use-view-consistency
```

Use `--max-steps 2` for a quick smoke run. Parameters are controlled only by `train.py`; model architecture is in `model.py`.
