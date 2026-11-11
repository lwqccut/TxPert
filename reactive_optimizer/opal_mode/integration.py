"""Single-device FP32 Lightning bridge to the pinned OPAL optimizer."""
import math
from pathlib import Path

import torch
from lightning.pytorch import Callback
import opal

from reactive_optimizer.controller import ReactiveController
from reactive_optimizer.predictor import ReactivePredictor
from reactive_optimizer.core import write_json
from . import UPSTREAM_VERSION, UPSTREAM_COMMIT
from .backend import QwenCodeBackend, validate_options


def check_version():
    if opal.__version__ != UPSTREAM_VERSION:
        raise RuntimeError(f'Install the pinned OPAL {UPSTREAM_VERSION} dependency from opal_mode/requirements.txt')


class OpalPredictor(ReactivePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opal_loss_sum = None
        self._opal_loss_weight = 0

    def _record_opal_training_loss(self, loss, batch_size):
        """Accumulate detached micro-batch losses until Lightning performs an update."""
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError('OPAL requires a scalar training loss for every micro-batch')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('OPAL requires a positive integer batch size')
        weighted = loss.detach() * batch_size
        self._opal_loss_sum = weighted if self._opal_loss_sum is None else self._opal_loss_sum + weighted
        self._opal_loss_weight += batch_size

    def pop_opal_training_loss(self):
        """Return the sample-weighted loss since the previous optimizer update."""
        if self._opal_loss_sum is None or self._opal_loss_weight < 1:
            raise RuntimeError('OPAL did not receive a training loss before the optimizer update')
        loss = self._opal_loss_sum / self._opal_loss_weight
        self._opal_loss_sum = None
        self._opal_loss_weight = 0
        return loss

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self._record_opal_training_loss(loss, int(batch.control.shape[0]))
        return loss

    def configure_opal(self, agent, options, output, keep=False, backend=None):
        check_version()
        validate_options(options)
        self.opal_agent, self.opal_options = agent, options
        self.opal_output, self.opal_keep = Path(output), keep
        # Also initialize when tests or downstream wrappers adapt an existing
        # ReactivePredictor instance into OpalPredictor after construction.
        if not hasattr(self, '_opal_loss_sum'):
            self._opal_loss_sum = None
            self._opal_loss_weight = 0
        self.opal_backend = backend

    def configure_optimizers(self):
        original = super().configure_optimizers()
        if isinstance(original, dict):
            raise ValueError('OPAL mode currently requires lr_scheduler.type=null and no warmup')
        steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(steps) or steps < 1:
            raise ValueError('OPAL needs a finite positive training budget')
        options = self.opal_options
        named = list(self.model.named_parameters())
        names = {id(p): n for n, p in named}
        specs = {'parameters': [{'name': name, 'shape': list(p.shape)} for name, p in named if p.requires_grad],
                 'groups': [{**{k: g[k] for k in ('name', 'lr', 'weight_decay', 'betas', 'eps')},
                             'params': [names[id(p)] for p in g['params']]} for g in original.param_groups],
                 'clip_norm': self.opal_agent['clip_norm'], 'total_steps': int(steps)}
        schedule = list(range(options['interval_steps'], int(steps), options['interval_steps']))
        # Reserve part of the total call cap for analysis-triggered wake-ups.
        schedule = schedule[:options['max_decisions']]
        backend = 'keep' if self.opal_keep else (self.opal_backend or QwenCodeBackend(self.opal_agent, options, specs))
        wrapped = opal.wrap(original, self.model, total_steps=int(steps), goal=options['goal'],
                            backend=backend, schedule=schedule, every=options['every'],
                            max_decisions=options['max_decisions'], blocking=options['blocking'],
                            trial_steps=options['trial_steps'], snapshot_limit_gb=options['snapshot_limit_gb'],
                            clip_grad_norm=self.opal_agent['clip_norm'],
                            share_code=options.get('share_code', False),
                            summarize=False, pause_on_nonfinite=False,
                            run_dir=self.opal_output / 'opal')
        write_json(self.opal_output / 'opal-settings.json', {'version': opal.__version__,
                   'upstream_commit': UPSTREAM_COMMIT, 'options': options, 'keep': self.opal_keep,
                   'parameter_specs': specs, 'decision_steps': schedule,
                   'share_code': options.get('share_code', False),
                   'selection_metric': options.get('selection_metric', 'pearson_delta')})
        return wrapped


class OpalAudit(ReactiveController):
    """Reuse RNG checkpointing only; bounded actions/feedback are not active here."""
    def __init__(self, agent, options, directory, prompt, keep=False):
        super().__init__(agent, directory, prompt, enabled=not keep)
        self.options = options
        self.identity['opal_options'] = options

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != 1 or str(trainer.precision) not in ('32', '32-true'):
            raise ValueError('This OPAL bridge supports one device and FP32 only')

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # OPAL's starting updater handles clipping, preventing double clipping.
        trainer.gradient_clip_val = 0.0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Lightning owns optimizer.step(), so OPAL's documented integration is
        # observe(loss) once per actual optimizer update. This hook runs after
        # gradient accumulation and before the wrapped optimizer is stepped.
        loss = pl_module.pop_opal_training_loss()
        target = optimizer
        if not callable(getattr(target, 'observe', None)):
            target = getattr(target, 'optimizer', target)
        observe = getattr(target, 'observe', None)
        if not callable(observe):
            raise RuntimeError('The active OPAL optimizer does not expose observe(loss)')
        observe(loss)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get('reactive_val_mse')
        if value is not None and math.isfinite(float(value)):
            self.val = {'mse': float(value), 'step': int(trainer.global_step)}
            trainer.optimizers[0].log(validation_mse=value, show_agent=True)
            pearson = trainer.callback_metrics.get('reactive_val_pearson_delta')
            if pearson is not None and math.isfinite(float(pearson)):
                self.val['pearson_delta'] = float(pearson)
                trainer.optimizers[0].log(validation_pearson_delta=pearson, show_agent=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step > self.last_step and step % self.options['checkpoint_every'] == 0:
            self.last_step = step
            folder = self.directory / 'opal-checkpoints'
            folder.mkdir(exist_ok=True)
            trainer.save_checkpoint(folder / f'step-{step:09d}.ckpt')

    def on_train_end(self, trainer, pl_module):
        trainer.optimizers[0].close()
        write_json(self.directory / 'summary.json', {'engine': 'official-opal',
                   'global_step': int(trainer.global_step), 'last_validation': self.val,
                   'selection_metric': self.options.get('selection_metric', 'pearson_delta'),
                   'keep': not self.enabled,
                   'note': 'Code acceptance/trial survival is not evidence of improved accuracy'})

    def on_exception(self, trainer, pl_module, exception):
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, opal.AgentOptimizer):
                optimizer.close(interrupted=True)
