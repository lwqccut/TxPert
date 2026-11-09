"""Run from repository root: python -m reactive_optimizer.train --help."""
import argparse
import copy
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .core import digest, validate_config, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('bounded', 'opal'), default='bounded')
    parser.add_argument('--opal-config', default=str(Path(__file__).with_name('opal_mode') / 'config.yaml'))
    parser.add_argument('--config', default='config-gat', help='Name in original configs/')
    parser.add_argument('--agent-config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--revision', help='Exact model-weight commit served by the LLM endpoint')
    parser.add_argument('--endpoint', help='OpenAI-compatible server /v1 URL')
    parser.add_argument('--output', required=True, help='New directory for this run, including resume branches')
    parser.add_argument('--baseline', action='store_true', help='Same grouping/logging without LLM actions')
    parser.add_argument('--resume', help='Trusted Lightning checkpoint; continues optimizer/controller state')
    parser.add_argument('--download-data', action='store_true', help='Run original Zenodo downloader first')
    parser.add_argument('--print-config', action='store_true', help='Validate/print configuration without model imports')
    args, overrides = parser.parse_known_args()
    agent = yaml.safe_load(Path(args.agent_config).read_text())
    if args.revision:
        agent['revision'] = args.revision
    if args.endpoint:
        agent['endpoint'] = args.endpoint
    validate_config(agent)
    opal_options = yaml.safe_load(Path(args.opal_config).read_text()) if args.engine == 'opal' else None
    url = urlsplit(agent['endpoint'])
    if url.scheme not in ('http', 'https') or url.username or url.password or url.query or url.fragment:
        parser.error('Endpoint must be HTTP(S), with no embedded credentials or query')
    if not args.baseline and not re.fullmatch(r'[0-9a-fA-F]{40}', str(agent.get('revision', ''))):
        parser.error('Live runs require --revision (the exact model commit used by your server)')
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
        cfg = compose(config_name=args.config, overrides=['mode=train', *overrides])
    config = OmegaConf.to_container(cfg, resolve=True)
    if config['mode'] != 'train' or config['model']['model_type'] != 'txpert':
        parser.error('This entrypoint requires mode=train and the original txpert model')
    if args.print_config:
        print(OmegaConf.to_yaml(OmegaConf.create({'txpert': config, 'agent': agent, 'engine': args.engine, 'opal': opal_options})))
        return
    os.chdir(ROOT)
    # Writable caches, without changing system/user configuration.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
    import torch
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from gspp.data.datamodule import PertDataModule
    from gspp.data.graphmodule import GSPGraph
    from gspp.utils import set_seed, download_files_from_zenodo
    from .predictor import ReactivePredictor
    from .controller import ReactiveController

    if args.engine == 'opal':
        from .opal_mode.integration import OpalPredictor, OpalAudit
        from .opal_mode.backend import validate_options
        validate_options(opal_options)
    trainer_args = copy.deepcopy(config.get('trainer', {}))
    if trainer_args.get('devices', 1) not in (1, [0]) or trainer_args.get('num_nodes', 1) != 1 or trainer_args.get('strategy', 'auto') != 'auto':
        parser.error('This entrypoint supports one device and strategy=auto only')
    if trainer_args.get('gradient_clip_algorithm', 'norm') != 'norm':
        parser.error('Reactive clipping requires gradient_clip_algorithm=norm')
    if trainer_args.get('gradient_clip_val') is not None:
        agent['clip_norm'] = float(trainer_args['gradient_clip_val'])
        validate_config(agent)
    if args.engine == 'opal' and str(trainer_args.get('precision', '32-true')) not in ('32', '32-true'):
        parser.error('OPAL mode currently supports FP32 only')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt_path = (Path(__file__).with_name('opal_mode') / 'prompt.txt' if args.engine == 'opal'
                   else Path(__file__).with_name('prompt.txt'))
    prompt = prompt_path.read_text()
    (output / 'prompt.txt').write_text(prompt)
    manifest = {
        'txpert': config, 'agent': agent, 'baseline': args.baseline,
        'engine': args.engine, 'opal': opal_options,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        'python': sys.version, 'torch': torch.__version__,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'prompt_sha256': digest(prompt_path),
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ('gspp', 'reactive_optimizer')
                          for p in (ROOT / folder).rglob('*.py')},
        'model_identity': 'Operator-declared weights revision and precision; server not independently attested',
        'resume': str(Path(args.resume).resolve()) if args.resume else None,
        'resume_sha256': digest(args.resume) if args.resume else None,
    }
    write_json(output / 'manifest.json', manifest)
    set_seed(config['seed'])
    if args.download_data:
        download_files_from_zenodo('15420279')
    dm = PertDataModule(**config['datamodule'])
    dm.prepare_data()
    graph = GSPGraph(pert2id=dm.pert2id, gene2id=dm.gene2id, **config['graph'])
    use_gpu = torch.cuda.is_available() and trainer_args.get('accelerator') != 'cpu'
    predictor_class = OpalPredictor if args.engine == 'opal' else ReactivePredictor
    model = predictor_class(
        input_dim=dm.input_dim, output_dim=dm.output_dim, adata_output_dim=dm.adata_output_dim,
        graph=graph, model_args=copy.deepcopy(config['model']), device='cuda' if use_gpu else 'cpu',
        lr=config.get('lr', 1e-3), min_lr=config.get('min_lr', 0.0),
        lr_scheduler_args=config.get('lr_scheduler') or {},
        weight_decay=config.get('weight_decay', 0.0),
        match_cntr_for_eval=config.get('match_cntr_for_eval', True),
    )
    if args.engine == 'opal':
        model.configure_opal(agent, opal_options, output, keep=args.baseline)
        controller = OpalAudit(agent, opal_options, output, prompt, keep=args.baseline)
    else:
        controller = ReactiveController(agent, output, prompt, enabled=not args.baseline)
    trainer_args.setdefault('accelerator', 'gpu' if use_gpu else 'cpu')
    trainer_args.setdefault('devices', 1)
    trainer_args['default_root_dir'] = str(output)
    trainer_args['gradient_clip_val'] = agent['clip_norm']
    trainer_args['gradient_clip_algorithm'] = 'norm'
    trainer_args['callbacks'] = [controller, ModelCheckpoint(
        dirpath=output / 'checkpoints', filename='{epoch:03d}-{step}',
        save_last=True, save_top_k=1, monitor='reactive_val_mse', mode='min')]
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.save_checkpoint(output / 'final.ckpt')


if __name__ == '__main__':
    main()
