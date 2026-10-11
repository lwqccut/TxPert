"""Ablation 01: single STRING graph, unchanged forward TxPert/Exphormer."""
from pathlib import Path
import argparse, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--limit-batches', type=int, default=0); args = p.parse_args()
    out = ROOT / 'forward_ablation' / 'runs' / '01_baseline'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'main.py', '--config-name=config-exphormer', 'mode=train',
           f'+trainer.max_epochs={args.epochs}', f'datamodule.batch_size={args.batch_size}',
           *( [f'+trainer.limit_train_batches={args.limit_batches}'] if args.limit_batches else [] ),
           f'+checkpoint_dir={out}']
    env = dict(os.environ, TXPERT_SKIP_DOWNLOAD='1')
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))
if __name__ == '__main__': main()
