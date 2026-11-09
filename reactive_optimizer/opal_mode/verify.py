"""Separate-process original-GAT keep parity check on synthetic data, no LLM/API."""
import argparse
from pathlib import Path
import os
import subprocess
import sys


def child(mode, steps, output):
    import torch
    import yaml
    from types import SimpleNamespace
    from reactive_optimizer.predictor import ReactivePredictor
    from reactive_optimizer.core import write_json
    torch.set_num_threads(1)
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / 'configs/config-gat.yaml').read_text())
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).numpy()
    graph = SimpleNamespace(graph_dict={'string': (edges, torch.ones(4).numpy(), 4)})
    model = ReactivePredictor(8, 8, 8, cfg['model'], graph=graph, lr=cfg['lr'])
    optimizer = model.configure_optimizers()
    if mode == 'keep':
        import opal
        optimizer = opal.wrap(optimizer, model.model, total_steps=steps, goal='Synthetic original-GAT keep parity',
                              backend='keep', schedule=list(range(2, steps, 2)), every=1,
                              clip_grad_norm=1.0, summarize=False, share_code=False, run_dir=output/'opal')
    batches = [(torch.randn(2, 8), torch.randn(2, 8)) for _ in range(steps)]
    losses = []
    for x, target in batches:
        optimizer.zero_grad()
        prediction, mean, logvar = model.model(x, [[0], [1]], torch.zeros(2, 1))
        loss, _ = model.model.loss(prediction, mean, logvar, target)
        loss.backward()
        if mode == 'keep':
            optimizer.step(loss=loss)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach()))
    if mode == 'keep':
        optimizer.close()
    torch.save(model.state_dict(), output/f'{mode}.pt')
    write_json(output/f'{mode}.json', {'losses': losses})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--child', choices=('plain', 'keep'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('steps must be >= 2')
    output = Path(args.output).resolve()
    if args.child:
        child(args.child, args.steps, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    for mode in ('plain', 'keep'):
        with (output/f'{mode}.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'reactive_optimizer.opal_mode.verify',
                            '--output', str(output), '--steps', str(args.steps), '--child', mode],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    import json
    import torch
    from reactive_optimizer.core import write_json
    a, b = [torch.load(output/f'{mode}.pt', weights_only=True) for mode in ('plain', 'keep')]
    losses = [json.loads((output/f'{mode}.json').read_text())['losses'] for mode in ('plain', 'keep')]
    result = {'steps': args.steps, 'losses_exact': losses[0] == losses[1],
              'weights_exact': a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a),
              'scope': 'CPU synthetic data, original GAT configuration; no real LLM or biology evaluation'}
    write_json(output/'result.json', result)
    print(json.dumps(result))
    if not result['losses_exact'] or not result['weights_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
