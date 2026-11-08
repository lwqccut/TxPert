"""Synchronous, single-device controller with durable audit records."""
import json
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback

from .core import ChatClient, validate_action, write_json


def optimizer_settings(optimizer):
    return [{k: g[k] for k in ('name', 'lr', 'weight_decay', 'betas', 'eps')}
            for g in optimizer.param_groups]


class ReactiveController(Callback):
    def __init__(self, cfg, directory, prompt, enabled=True, client=None):
        self.cfg, self.directory = cfg, Path(directory)
        self.enabled = enabled
        self.identity = {'config': cfg, 'enabled': enabled,
                         'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        self.client = client or ChatClient(cfg, prompt)
        self.clip_norm = cfg['clip_norm']
        self.last_step = 0
        self.counts = {'attempts': 0, 'accepted': 0, 'changed': 0, 'failed': 0}
        self.history = []
        self.grad_stats = {}
        self.val = None
        self.pending_rng = None

    def state_dict(self):
        return {'identity': self.identity, 'clip_norm': self.clip_norm, 'last_step': self.last_step,
                'history': self.history, 'val': self.val, 'counts': self.counts}

    def load_state_dict(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Resume requires the same controller config, model identity, prompt and baseline mode')
        for key in ('clip_norm', 'last_step', 'history', 'val', 'counts'):
            setattr(self, key, state[key])

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['reactive_rng'] = {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.pending_rng = checkpoint.get('reactive_rng')

    def on_train_start(self, trainer, pl_module):
        trainer.gradient_clip_val = self.clip_norm
        if self.pending_rng:
            random.setstate(self.pending_rng['python'])
            np.random.set_state(self.pending_rng['numpy'])
            torch.set_rng_state(self.pending_rng['torch'].cpu())
            if self.pending_rng['cuda']:
                torch.cuda.set_rng_state_all([s.cpu() for s in self.pending_rng['cuda']])
            self.pending_rng = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Lightning calls this after AMP unscaling and before gradient clipping.
        self.grad_stats = {}
        for group in optimizer.param_groups:
            params = group['params']
            grad_sq = sum(p.grad.detach().float().square().sum() for p in params if p.grad is not None)
            weight_sq = sum(p.detach().float().square().sum() for p in params)
            grad_norm = float(torch.as_tensor(grad_sq).sqrt())
            weight_norm = float(torch.as_tensor(weight_sq).sqrt())
            self.grad_stats[group['name']] = {
                'grad_norm': grad_norm if math.isfinite(grad_norm) else None,
                'weight_norm': weight_norm if math.isfinite(weight_norm) else None,
            }

    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get('reactive_val_mse')
        if not trainer.sanity_checking and value is not None and math.isfinite(float(value)):
            self.val = {'mse': float(value), 'step': int(trainer.global_step)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step <= self.last_step or step % self.cfg['interval_steps']:
            return
        self.last_step = step
        optimizer = trainer.optimizers[0]
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        loss = float(loss.detach()) if torch.is_tensor(loss) else None
        observation = {
            'step': step, 'epoch': int(trainer.current_epoch),
            'train_loss': loss if loss is not None and math.isfinite(loss) else None,
            'validation': self.val, 'groups': optimizer_settings(optimizer),
            'gradients': self.grad_stats, 'clip_norm': self.clip_norm,
            'bounds': {k: self.cfg[k] for k in ('lr_factor_min', 'lr_factor_max', 'lr_min', 'lr_max',
                                               'weight_decay_max', 'clip_min', 'clip_max')},
            'history': list(self.history),
        }
        # mkdtemp permits branching/resuming without overwriting any old decisions.
        import tempfile
        event = Path(tempfile.mkdtemp(prefix=f'step-{step:09d}-', dir=self.directory))
        payload = self.client.payload(observation)
        write_json(event / 'request.json', payload)
        trainer.save_checkpoint(event / 'before.ckpt')
        record = {'step': step, 'enabled': self.enabled, 'before': optimizer_settings(optimizer),
                  'clip_before': self.clip_norm, 'status': 'baseline'}
        started = time.monotonic()
        if self.enabled:
            self.counts['attempts'] += 1
            try:
                response = self.client.complete(payload)
                write_json(event / 'response.json', response)
                action = json.loads(response['choices'][0]['message']['content'])
                group, field, value = validate_action(action, optimizer, self.cfg)
                if field == 'clip_norm':
                    self.clip_norm = value
                    trainer.gradient_clip_val = value
                elif group is not None:
                    group[field] = value
                record.update(status='accepted', action=action)
                self.counts['accepted'] += 1
                self.counts['changed'] += int(field is not None)
            except Exception as error:
                # No exception text: HTTP errors may contain credential-bearing URLs.
                record.update(status='rejected_or_unavailable', error_type=type(error).__name__)
                self.counts['failed'] += 1
        record.update(elapsed_seconds=time.monotonic() - started,
                      after=optimizer_settings(optimizer), clip_after=self.clip_norm)
        write_json(event / 'decision.json', record)
        self.history.append({'step': step, 'train_loss': observation['train_loss'],
                             'validation': self.val, 'status': record['status'],
                             'action': record.get('action')})
        self.history = self.history[-self.cfg['history_size']:]
        trainer.save_checkpoint(event / 'after.ckpt')
        print(f"Reactive step {step}: {record['status']} ({event.name})", flush=True)

    def on_train_end(self, trainer, pl_module):
        write_json(self.directory / 'summary.json', {
            'global_step': int(trainer.global_step), 'enabled': self.enabled,
            'counts_including_resumed_history': self.counts,
            'last_validation': self.val,
            'note': 'Accepted actions do not establish improved biological prediction.',
        })
