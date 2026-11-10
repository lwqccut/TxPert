"""Synchronous, single-device controller with durable audit records."""
import json
import copy
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback

from .core import ChatClient, validate_action, write_json
from .evidence import Evidence, finite


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
        self.counts = {'attempts': 0, 'accepted': 0, 'changed': 0, 'held': 0, 'no_op': 0, 'failed': 0}
        self.evidence = Evidence(cfg)
        self.history = []
        self.grad_stats = {}
        self.val = None
        self.pending_rng = None
        self.trial = None
        self.feedback_history = []
        self.cooldown_until = 0
        self.feedback_counts = {'kept': 0, 'rolled_back': 0}

    def state_dict(self):
        return {'feedback_version': 2, 'evidence': self.evidence.state(), 'trial': self.trial,
                'feedback_history': self.feedback_history, 'cooldown_until': self.cooldown_until,
                'feedback_counts': self.feedback_counts, 'identity': self.identity, 'clip_norm': self.clip_norm, 'last_step': self.last_step,
                'history': self.history, 'val': self.val, 'counts': self.counts}

    def load_state_dict(self, state):
        if state.get('feedback_version') != 2:
            raise ValueError('Checkpoint predates evidence/Pearson feedback; start a new experiment')
        if state['identity'] != self.identity:
            raise ValueError('Resume requires the same controller config, model identity, prompt and baseline mode')
        self.evidence.restore(state['evidence'])
        for key in ('clip_norm', 'last_step', 'history', 'val', 'counts', 'trial',
                    'feedback_history', 'cooldown_until', 'feedback_counts'):
            setattr(self, key, state[key])

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['reactive_rng'] = {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.pending_rng = checkpoint.get('reactive_rng')

    def on_fit_start(self, trainer, pl_module):
        # Restoring an old LR must not undo an independent scheduler's update.
        if self.enabled and self.cfg['feedback_enabled'] and trainer.lr_scheduler_configs:
            raise ValueError('Feedback mode requires no LR scheduler; disable feedback or the scheduler')

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
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get('reactive_val_mse')
        if value is None:
            return
        step, mse = int(trainer.global_step), float(value)
        finite = math.isfinite(mse) and mse >= 0
        pearson = trainer.callback_metrics.get('reactive_val_pearson_delta')
        pearson = float(pearson) if pearson is not None else None
        pearson_valid = pearson is not None and math.isfinite(pearson) and -1 <= pearson <= 1
        needs_pearson = self.cfg['feedback_metric'] == 'pearson_delta'
        valid = finite and (pearson_valid or not needs_pearson)
        measurement = {'mse': mse if finite else None, 'step': step,
                       'pearson_delta': pearson if pearson_valid else None}
        if not self.evidence.validate(measurement):
            return
        self.val = measurement if valid else None
        validation_dir = self.directory / 'validation'
        validation_dir.mkdir(exist_ok=True)
        write_json(validation_dir / f'step-{step:09d}.json', {
            **measurement, 'valid_for_decision': valid, 'validation_id': self.evidence.validation_id,
            'hold_streak': self.evidence.hold_streak,
            'next_routine_validation_id': self.evidence.next_validation_id,
            'evidence': self.evidence.report(),
        })
        trial = self.trial
        # Same-step validation has seen no training using the proposed settings.
        if trial is None or step <= trial['last_evaluation_step']:
            return
        if valid and step - trial['step'] < self.cfg['feedback_min_steps']:
            return
        trial['last_evaluation_step'] = step
        trial['evaluations'] += 1
        relative_change = ((mse - trial['baseline']['mse']) /
                           max(abs(trial['baseline']['mse']), 1e-12)) if finite else None
        status = 'pending'
        pearson_gain = (pearson-trial['baseline']['pearson_delta']) if needs_pearson and valid else None
        if not valid:
            status = 'rollback_nonfinite'
        else:
            bad = relative_change >= self.cfg['feedback_regression']
            if needs_pearson:
                bad = bad or pearson_gain <= -self.cfg['feedback_pearson_regression']
                good = pearson_gain >= self.cfg['feedback_pearson_improvement'] and not bad
            else:
                good = relative_change <= -self.cfg['feedback_improvement']
            trial['good'] = trial['good'] + 1 if good else 0
            trial['bad'] = trial['bad'] + 1 if bad else 0
            if trial['bad'] >= self.cfg['feedback_patience']:
                status = 'rollback_regression'
            elif trial['good'] >= self.cfg['feedback_patience']:
                status = 'kept'
            elif trial['evaluations'] >= self.cfg['feedback_max_evaluations']:
                status = 'rollback_inconclusive'
        record = {'trial_id': trial['id'], 'action': trial['action'],
                  'action_step': trial['step'], 'evaluation_step': step,
                  'baseline': trial['baseline'], 'mse': mse if finite else None,
                  'pearson_delta': pearson if pearson_valid else None, 'pearson_gain': pearson_gain,
                  'feedback_metric': self.cfg['feedback_metric'],
                  'relative_change': relative_change, 'status': status,
                  'evaluations': trial['evaluations'], 'good': trial['good'], 'bad': trial['bad'],
                  'rollback_scope': 'Changed hyperparameter only; weights and Adam moments are not rewound'}
        if status.startswith('rollback_'):
            if trial['field'] == 'clip_norm':
                self.clip_norm = trial['old_value']
                trainer.gradient_clip_val = self.clip_norm
            else:
                group = next(g for g in trainer.optimizers[0].param_groups if g['name'] == trial['group'])
                group[trial['field']] = trial['old_value']
            self.feedback_counts['rolled_back'] += 1
        elif status == 'kept':
            self.feedback_counts['kept'] += 1
        if status != 'pending':
            self.trial = None
            self.cooldown_until = step + self.cfg['feedback_cooldown_steps']
        record['settings_after'] = optimizer_settings(trainer.optimizers[0])
        record['clip_after'] = self.clip_norm
        self.feedback_history.append(record)
        self.feedback_history = self.feedback_history[-self.cfg['history_size']:]
        folder = self.directory / 'feedback'
        folder.mkdir(exist_ok=True)
        stem = f"{trial['id']}-eval-{step:09d}"
        write_json(folder / (stem + '.json'), record)
        trainer.save_checkpoint(folder / (stem + '.ckpt'))
        print(f'Reactive feedback step {step}: {status}', flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        loss = float(loss.detach()) if torch.is_tensor(loss) else None
        # One sample per optimizer update, including with gradient accumulation.
        if step <= 0 or step <= self.evidence.last_observed_step:
            return
        self.evidence.observe(step, loss, self.grad_stats)
        if step <= self.last_step:
            return
        if self.cfg['decision_trigger'] == 'interval':
            if step % self.cfg['interval_steps']:
                return
            trigger = {'kind': 'fixed_interval_ablation'}
        else:
            if step - self.last_step < self.cfg['interval_steps']:
                return
            trigger = self.evidence.trigger()
            if trigger is None:
                self.evidence.skip('no_new_evidence_or_hold_backoff')
                return
        optimizer = trainer.optimizers[0]
        if self.enabled and (self.cfg['feedback_enabled'] or self.cfg['decision_trigger'] == 'evidence'):
            # No stacked interventions, no proposals without a recent finite reference.
            if (self.trial is not None or step < self.cooldown_until or self.val is None
                    or step - self.val['step'] > self.cfg['feedback_baseline_max_age_steps']):
                self.evidence.skip('pending_trial_cooldown_or_missing_fresh_baseline')
                return
        self.last_step = step
        self.evidence.consume()
        budget = getattr(trainer, 'estimated_stepping_batches', None)
        observation = {
            'step': step, 'epoch': int(trainer.current_epoch),
            'train_loss': loss if loss is not None and math.isfinite(loss) else None,
            'validation': self.val, 'groups': optimizer_settings(optimizer),
            'gradients': self.grad_stats, 'clip_norm': self.clip_norm,
            'bounds': {k: self.cfg[k] for k in ('lr_factor_min', 'lr_factor_max', 'lr_min', 'lr_max',
                                               'weight_decay_max', 'clip_min', 'clip_max')},
            'history': list(self.history),
            'feedback_history': list(self.feedback_history),
            'trigger': trigger, 'evidence': self.evidence.report(),
            'objective': {'primary': self.cfg['feedback_metric'], 'mse_regression_guard': self.cfg['feedback_regression'],
                          'pearson_gain_required': self.cfg['feedback_pearson_improvement'],
                          'min_trial_steps': self.cfg['feedback_min_steps'],
                          'consecutive_validations_required': self.cfg['feedback_patience']},
            'training_budget': {'total_steps': int(budget) if finite(budget) else None,
                                'remaining_steps': max(0, int(budget)-step) if finite(budget) else None},
        }
        # mkdtemp permits branching/resuming without overwriting any old decisions.
        import tempfile
        event = Path(tempfile.mkdtemp(prefix=f'step-{step:09d}-', dir=self.directory))
        payload = self.client.payload(observation)
        write_json(event / 'request.json', payload)
        trainer.save_checkpoint(event / 'before.ckpt')
        record = {'step': step, 'enabled': self.enabled, 'before': optimizer_settings(optimizer),
                  'clip_before': self.clip_norm, 'status': 'baseline', 'changed': False, 'trigger': trigger}
        started = time.monotonic()
        if self.enabled:
            self.counts['attempts'] += 1
            try:
                response = self.client.complete(payload)
                write_json(event / 'response.json', response)
                action = json.loads(response['choices'][0]['message']['content'])
                group, field, value = validate_action(action, optimizer, self.cfg)
                old_value = self.clip_norm if field == 'clip_norm' else (group[field] if group is not None else None)
                if field == 'clip_norm':
                    self.clip_norm = value
                    trainer.gradient_clip_val = value
                elif group is not None:
                    group[field] = value
                changed = field is not None and value != old_value
                if changed and self.cfg['feedback_enabled']:
                    self.trial = {
                        'id': event.name, 'step': step, 'action': action,
                        'baseline': copy.deepcopy(self.val), 'group': action.get('group'),
                        'field': field, 'old_value': old_value, 'new_value': value,
                        'evaluations': 0, 'good': 0, 'bad': 0, 'last_evaluation_step': step,
                    }
                effect = 'applied' if changed else ('hold' if action['action'] == 'hold' else 'no_op')
                record.update(status='accepted', action=action, changed=changed, decision_effect=effect,
                              pending_feedback=self.trial is not None)
                self.counts['accepted'] += 1
                self.counts['changed'] += int(changed)
                self.counts['held'] += int(effect == 'hold')
                self.counts['no_op'] += int(effect == 'no_op')
                self.evidence.outcome(unchanged=not changed)
            except Exception as error:
                # No exception text: HTTP errors may contain credential-bearing URLs.
                record.update(status='rejected_or_unavailable', error_type=type(error).__name__)
                self.counts['failed'] += 1
        record.update(elapsed_seconds=time.monotonic() - started,
                      after=optimizer_settings(optimizer), clip_after=self.clip_norm,
                      hold_streak=self.evidence.hold_streak,
                      next_routine_validation_id=self.evidence.next_validation_id)
        write_json(event / 'decision.json', record)
        self.history.append({'step': step, 'train_loss': observation['train_loss'],
                             'validation': self.val, 'status': record['status'],
                             'action': record.get('action')})
        self.history = self.history[-self.cfg['history_size']:]
        trainer.save_checkpoint(event / 'after.ckpt')
        print(f"Reactive step {step}: {record['status']}, effect={record.get('decision_effect', 'none')}, "
              f"changed={record['changed']}, hold_streak={self.evidence.hold_streak} ({event.name})", flush=True)

    def on_train_end(self, trainer, pl_module):
        write_json(self.directory / 'summary.json', {
            'global_step': int(trainer.global_step), 'enabled': self.enabled,
            'counts_including_resumed_history': self.counts,
            'last_validation': self.val,
            'feedback_counts': self.feedback_counts,
            'pending_trial': self.trial,
            'evidence_state': self.evidence.state(),
            'intervention_observed': self.counts['changed'] > 0,
            'warning': ('LLM was called but never changed training settings; this run does not demonstrate an intervention.'
                        if self.counts['attempts'] and not self.counts['changed'] else None),
            'note': 'Accepted actions do not establish improved biological prediction.',
        })
