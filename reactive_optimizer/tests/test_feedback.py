"""Controlled validation sequences test policy; not evidence of biological benefit."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from reactive_optimizer.controller import ReactiveController
from reactive_optimizer.core import ChatClient, validate_config

BASE = Path(__file__).resolve().parents[1]
PROMPT = (BASE / 'prompt.txt').read_text()
CONFIG = yaml.safe_load((BASE / 'config.yaml').read_text())


class Client(ChatClient):
    action = {'action': 'scale_lr', 'group': 'graph_encoder', 'factor': 0.8, 'reason': 'test'}

    def complete(self, payload):
        return {'choices': [{'message': {'content': json.dumps(self.action)}}]}


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {**CONFIG, 'interval_steps': 2, 'feedback_min_steps': 2,
                    'feedback_baseline_max_age_steps': 10, 'feedback_cooldown_steps': 4,
                    'feedback_metric': 'mse', 'decision_trigger': 'interval'}
        self.client = Client(self.cfg, PROMPT)
        self.controller = ReactiveController(self.cfg, self.tmp.name, PROMPT, client=self.client)
        self.parameter = torch.nn.Parameter(torch.ones(2))
        self.opt = torch.optim.AdamW([{'name': 'graph_encoder', 'params': [self.parameter]}], lr=0.001)
        self.parameter.sum().backward()
        self.opt.step()  # Initialize real Adam moments to check rollback preserves them.
        self.trainer = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[self.opt],
                                       sanity_checking=False, callback_metrics={}, gradient_clip_val=1.0,
                                       lr_scheduler_configs=[], save_checkpoint=lambda p: Path(p).touch())

    def validate(self, step, mse):
        self.trainer.global_step = step
        self.trainer.callback_metrics['reactive_val_mse'] = torch.tensor(mse, dtype=torch.float64)
        self.controller.on_validation_end(self.trainer, None)

    def batch(self, step):
        self.trainer.global_step = step
        self.controller.on_train_batch_end(self.trainer, None, {'loss': torch.tensor(1.)}, None, 0)

    def start_trial(self):
        self.validate(1, 1.0)
        self.batch(2)
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.0008)
        self.assertIsNotNone(self.controller.trial)

    def test_requires_baseline_and_rejects_stale_reference(self):
        with patch.object(self.client, 'complete', side_effect=AssertionError('unexpected call')):
            self.batch(2)
            self.validate(3, 1.0)
            self.batch(20)
        self.assertEqual(self.controller.counts['attempts'], 0)

    def test_keeps_only_after_fresh_repeated_improvement(self):
        self.start_trial()
        self.validate(2, 0.1)  # Same step as intervention: no training under it yet.
        self.validate(3, 0.1)  # Too few steps.
        self.assertEqual(self.controller.trial['evaluations'], 0)
        self.validate(4, 0.98)
        self.validate(4, 0.97)  # Duplicate validation must not increase patience.
        self.assertEqual(self.controller.trial['good'], 1)
        with patch.object(self.client, 'complete', side_effect=AssertionError('stacked action')):
            self.batch(4)
        self.validate(6, 0.97)
        self.assertIsNone(self.controller.trial)
        self.assertEqual(self.controller.feedback_counts['kept'], 1)
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.0008)
        with patch.object(self.client, 'complete', side_effect=AssertionError('cooldown')):
            self.batch(8)
        self.batch(10)
        self.assertEqual(self.controller.counts['attempts'], 2)
        request = json.loads(sorted(Path(self.tmp.name).glob('step-000000010-*/request.json'))[0].read_text())
        self.assertIn('kept', request['messages'][1]['content'])

    def test_regression_restores_only_changed_setting_not_weights_or_moments(self):
        self.start_trial()
        weights = self.parameter.detach().clone()
        moments = self.opt.state[self.parameter]['exp_avg'].clone()
        self.validate(4, 1.08)
        self.assertIsNotNone(self.controller.trial)
        self.validate(6, 1.09)
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.001)
        self.assertTrue(torch.equal(weights, self.parameter))
        self.assertTrue(torch.equal(moments, self.opt.state[self.parameter]['exp_avg']))
        self.assertEqual(self.controller.feedback_history[-1]['status'], 'rollback_regression')
        self.assertTrue(list((Path(self.tmp.name)/'feedback').glob('*.ckpt')))

    def test_inconclusive_and_oscillating_results_expire(self):
        self.start_trial()
        for step, mse in [(4, 0.98), (6, 1.0), (8, 0.98), (10, 1.0)]:
            self.validate(step, mse)
        self.assertEqual(self.controller.feedback_history[-1]['status'], 'rollback_inconclusive')
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.001)

    def test_nonfinite_validation_rolls_back_without_waiting(self):
        self.start_trial()
        self.validate(3, float('nan'))
        self.assertIsNone(self.controller.val)
        self.assertEqual(self.controller.feedback_history[-1]['status'], 'rollback_nonfinite')
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.001)

    def test_clip_and_decay_rollback(self):
        for action, field, expected in [
            ({'action':'set_clip_norm','value':2.0,'reason':'test'}, 'clip_norm', 1.0),
            ({'action':'set_weight_decay','group':'graph_encoder','value':0.02,'reason':'test'}, 'weight_decay', 0.01),
        ]:
            with self.subTest(field=field):
                self.controller = ReactiveController(self.cfg, self.tmp.name, PROMPT, client=self.client)
                self.client.action = action
                self.validate(1, 1.0)
                self.batch(2)
                self.validate(4, 1.1)
                self.validate(6, 1.2)
                actual = self.trainer.gradient_clip_val if field == 'clip_norm' else self.opt.param_groups[0][field]
                self.assertAlmostEqual(actual, expected)

    def test_pending_trial_survives_checkpoint_state_restore(self):
        self.start_trial()
        self.validate(4, 1.1)
        state = copy.deepcopy(self.controller.state_dict())
        restored = ReactiveController(self.cfg, self.tmp.name, PROMPT, client=self.client)
        restored.load_state_dict(state)
        self.controller = restored
        self.validate(6, 1.1)
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], 0.001)
        self.assertEqual(self.controller.feedback_counts['rolled_back'], 1)

    def test_sanity_validation_does_not_supply_reference(self):
        self.trainer.sanity_checking = True
        self.validate(0, 1.0)
        self.assertIsNone(self.controller.val)

    def test_scheduler_conflict_and_config_validation(self):
        self.trainer.lr_scheduler_configs = [object()]
        with self.assertRaises(ValueError):
            self.controller.on_fit_start(self.trainer, None)
        for changes in [{'feedback_patience': 5}, {'feedback_improvement': 0},
                        {'feedback_enabled': 'true'}, {'feedback_min_steps': 0}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.cfg, **changes})

    def test_end_of_training_keeps_unresolved_trial_explicit(self):
        self.start_trial()
        self.controller.on_train_end(self.trainer, None)
        summary = json.loads((Path(self.tmp.name)/'summary.json').read_text())
        self.assertIsNotNone(summary['pending_trial'])
        self.assertEqual(summary['feedback_counts'], {'kept':0, 'rolled_back':0})


if __name__ == '__main__':
    unittest.main()
