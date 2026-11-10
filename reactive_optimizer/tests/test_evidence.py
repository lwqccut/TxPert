"""Evidence scheduling and metric feedback tests; scripted actions are not an LLM efficacy study."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
import anndata
from lightning import Trainer

from reactive_optimizer.core import ChatClient, validate_config
from reactive_optimizer.controller import ReactiveController
from reactive_optimizer.evidence import Evidence
from reactive_optimizer.validation import ValidationDelta, control_means
from reactive_optimizer.tests.test_reactive import TxPertIntegrationTests as _Base

# Importing TestCase classes into a test module would make unittest discover them twice.
model_fixture, loader_fixture = _Base.model, _Base.loaders
del _Base
ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT/'config.yaml').read_text())
PROMPT = (ROOT/'prompt.txt').read_text()


class Client(ChatClient):
    def __init__(self, cfg, action=None):
        super().__init__(cfg, PROMPT)
        self.action = action or {'action': 'hold', 'reason': 'fixture: await more evidence'}
        self.observations = []

    def complete(self, payload):
        self.observations.append(json.loads(payload['messages'][1]['content']))
        return {'choices': [{'message': {'content': json.dumps(self.action)}}]}


class EvidenceTests(unittest.TestCase):
    def setup_controller(self, changes=None, action=None):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = {**CONFIG, 'interval_steps': 2, 'trend_window': 3, 'trend_sample_every': 1,
                    'feedback_min_steps': 2, 'feedback_cooldown_steps': 0,
                    'feedback_baseline_max_age_steps': 100, **(changes or {})}
        self.client = Client(self.cfg, action)
        self.c = ReactiveController(self.cfg, self.root, PROMPT, client=self.client)
        self.opt = torch.optim.AdamW([{'name': 'graph_encoder', 'params': [torch.nn.Parameter(torch.ones(2))]}], lr=0.001)
        self.t = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[self.opt], sanity_checking=False,
                                 callback_metrics={}, save_checkpoint=lambda p: Path(p).touch(),
                                 gradient_clip_val=1., estimated_stepping_batches=100)

    def val(self, step, mse=1., pearson=0.5):
        self.t.global_step = step
        self.t.callback_metrics = {'reactive_val_mse': torch.tensor(mse, dtype=torch.float64)}
        if pearson is not None:
            self.t.callback_metrics['reactive_val_pearson_delta'] = torch.tensor(pearson, dtype=torch.float64)
        self.c.on_validation_end(self.t, None)

    def batch(self, step, loss=1.):
        self.t.global_step = step
        self.c.on_train_batch_end(self.t, None, {'loss': torch.tensor(loss)}, None, 0)

    def test_same_validation_not_reused_even_when_feedback_disabled(self):
        self.setup_controller({'feedback_enabled': False})
        self.val(1); self.batch(2)
        self.batch(4); self.batch(6)
        self.val(1); self.batch(8)
        self.assertEqual(len(self.client.observations), 1)
        self.val(9); self.batch(10)
        self.assertEqual(len(self.client.observations), 2)
        self.assertEqual(self.client.observations[-1]['trigger']['kind'], 'fresh_validation')

    def test_new_validation_waits_for_optimizer_boundary_not_microbatch(self):
        self.setup_controller()
        self.batch(2)  # Optimizer update happened before validation; no decision yet.
        self.val(2)
        self.batch(2)  # A microbatch with no additional optimizer update.
        self.assertEqual(self.c.counts['attempts'], 0)
        self.batch(3)
        self.assertEqual(self.c.counts['attempts'], 1)

    def test_pending_trial_blocks_event_and_does_not_consume_it(self):
        self.setup_controller(action={'action':'scale_lr','group':'graph_encoder','factor':.9,'reason':'fixture'})
        self.val(1)
        for step in (1,2,3): self.batch(step)
        for step in (4,5,6): self.batch(step, loss=3.)
        self.assertEqual(self.c.counts['attempts'], 1)
        self.assertGreater(self.c.evidence.event_id, self.c.evidence.used_event_id)

    def test_hold_backoff_is_bounded_and_auditable(self):
        self.setup_controller()
        calls = []
        for i in range(1, 25):
            before = self.c.counts['attempts']
            self.val(2*i-1, pearson=.5+i*.001); self.batch(2*i)
            if self.c.counts['attempts'] > before:
                calls.append(i)
        self.assertEqual(calls, [1, 2, 4, 8, 16, 24])
        self.assertEqual(self.c.counts['held'], 6)
        self.assertEqual(self.c.counts['changed'], 0)
        self.c.on_train_end(self.t, None)
        summary = json.loads((self.root/'summary.json').read_text())
        self.assertFalse(summary['intervention_observed'])
        self.assertIn('never changed', summary['warning'])
        event = json.loads(next(self.root.glob('step-000000002-*/decision.json')).read_text())
        self.assertFalse(event['changed']); self.assertEqual(event['decision_effect'], 'hold')

    def test_validation_regression_bypasses_hold_backoff_once(self):
        self.setup_controller()
        self.val(1); self.batch(2)
        self.val(3); self.batch(4)
        self.assertEqual(self.c.evidence.next_validation_id, 4)
        self.val(5, pearson=.47); self.batch(6)
        self.assertEqual(len(self.client.observations), 3)
        self.assertEqual(self.client.observations[-1]['trigger']['kind'], 'validation_regression')
        self.batch(8)
        self.assertEqual(len(self.client.observations), 3)

    def test_sustained_event_can_reuse_validation_but_not_repeat_episode(self):
        self.setup_controller()
        self.val(1)
        for step in (1, 2, 3): self.batch(step)
        for step in (4, 5, 6): self.batch(step, loss=3.)
        self.assertEqual(len(self.client.observations), 2)
        self.assertEqual(self.client.observations[-1]['trigger']['kind'], 'sustained_training_event')
        self.batch(7, loss=3.); self.batch(8, loss=3.)
        self.assertEqual(len(self.client.observations), 2)

    def test_single_noisy_batch_does_not_wake_agent(self):
        self.setup_controller()
        self.val(1)
        for step, loss in enumerate([1., 1., 1., 8., 1., 1., 1., 1.], 1):
            self.batch(step, loss)
        self.assertEqual(len(self.client.observations), 1)

    def test_checkpoint_preserves_backoff_windows_and_consumed_evidence(self):
        self.setup_controller()
        self.val(1); self.batch(2)
        self.val(3); self.batch(4)
        state = copy.deepcopy(self.c.state_dict())
        restored = ReactiveController(self.cfg, self.root, PROMPT, client=self.client)
        restored.load_state_dict(state); self.c = restored
        self.val(5); self.batch(6)
        self.assertEqual(len(self.client.observations), 2)
        self.val(7); self.batch(8)
        self.assertEqual(len(self.client.observations), 3)
        self.assertEqual(self.client.observations[-1]['evidence']['hold_streak'], 2)

    def test_missing_pearson_blocks_new_trial(self):
        self.setup_controller()
        self.val(1, pearson=None); self.batch(2)
        self.assertEqual(self.c.counts['attempts'], 0)

    def test_windows_include_variability_slope_and_do_not_count_microbatches(self):
        e = Evidence({**CONFIG, 'trend_window': 2, 'trend_sample_every': 1})
        for step in range(1, 5):
            e.observe(step, float(step), {'graph_encoder': {'grad_norm': float(step), 'weight_norm': 2.}})
            e.observe(step, 999., {})
        report = e.report()['training_windows']
        self.assertEqual(report['loss']['previous']['mean'], 1.5)
        self.assertEqual(report['loss']['recent']['mean'], 3.5)
        self.assertEqual(report['loss']['recent']['std'], .5)
        self.assertEqual(report['loss']['recent']['slope_per_step'], 1.)
        self.assertEqual(report['grad_weight_ratio:graph_encoder']['recent']['mean'], 1.75)

    def test_noop_action_is_not_counted_as_intervention(self):
        self.setup_controller(action={'action':'scale_lr','group':'graph_encoder','factor':1.,'reason':'fixture'})
        self.val(1); self.batch(2)
        self.assertEqual(self.c.counts['no_op'], 1)
        self.assertEqual(self.c.counts['changed'], 0)
        self.assertIsNone(self.c.trial)
        self.assertEqual(self.c.evidence.hold_streak, 1)

    def test_pearson_feedback_keeps_gain_and_rejects_metric_tradeoffs(self):
        action = {'action':'scale_lr','group':'graph_encoder','factor':.9,'reason':'fixture hypothesis'}
        cases = [(1., .52, 'kept'), (.8, .47, 'rollback_regression'),
                 (1.1, .52, 'rollback_regression'), (.8, .50, 'rollback_inconclusive')]
        for mse, pearson, expected in cases:
            with self.subTest(expected=expected, mse=mse):
                self.setup_controller(action=action)
                self.val(1); self.batch(2)
                for step in (4, 6, 8, 10):
                    if self.c.trial is not None: self.val(step, mse, pearson)
                self.assertEqual(self.c.feedback_history[-1]['status'], expected)
                self.assertAlmostEqual(self.opt.param_groups[0]['lr'], .0009 if expected=='kept' else .001)

    def test_invalid_pearson_rolls_back_without_waiting(self):
        self.setup_controller(action={'action':'scale_lr','group':'graph_encoder','factor':.9,'reason':'fixture'})
        self.val(1); self.batch(2)
        self.val(3, pearson=float('nan'))
        self.assertEqual(self.c.feedback_history[-1]['status'], 'rollback_nonfinite')
        self.assertAlmostEqual(self.opt.param_groups[0]['lr'], .001)

    def test_new_config_limits_and_old_checkpoint_rejected(self):
        for changes in [{'trend_window':1}, {'decision_trigger':'always'}, {'event_loss_ratio':1},
                        {'event_fraction':0}, {'hold_backoff_max_validations':0}, {'feedback_metric':'test'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**CONFIG, **changes})
        self.setup_controller()
        with self.assertRaisesRegex(ValueError, 'new experiment'):
            self.c.load_state_dict({'feedback_version':1})


class MetricTests(unittest.TestCase):
    def test_streaming_matches_original_pearson_delta_grouping_and_batch_means(self):
        from gspp.metrics import pearson_delta
        controls = {('K562','a'): torch.tensor([1., 2., 4.], dtype=torch.float64),
                    ('K562','b'): torch.tensor([2., 1., 3.], dtype=torch.float64)}
        pred = torch.tensor([[2.,3.,3.],[4.,3.,2.],[1.,3.,5.],[2.,5.,4.]], dtype=torch.float64)
        truth = torch.tensor([[2.,4.,2.],[5.,2.,2.],[1.,4.,6.],[3.,4.,5.]], dtype=torch.float64)
        conds, batches = ['p','q','p','p'], ['a','a','b','a']
        metric = ValidationDelta(controls)
        for indices in ([0,1], [2,3]):
            batch = SimpleNamespace(pert_cond_names=[conds[i] for i in indices], pert_dose=['1']*len(indices),
                                    cell_types=['K562']*len(indices), experimental_batches=[batches[i] for i in indices])
            metric.update(pred[indices], truth[indices], batch)
        expected = []
        for condition in ('p','q'):
            ids = [i for i,c in enumerate(conds) if c==condition]
            cntr = torch.stack([controls[('K562',batches[i])] for i in ids]).numpy()
            expected.append(pearson_delta(pred[ids].numpy(), truth[ids].numpy(), 'K562', cntr, None, condition))
        score, groups, undefined = metric.compute()
        self.assertAlmostEqual(score, np.mean(expected), places=12)
        self.assertEqual((groups, undefined), (2,0))
        # It must not accidentally use per-cell correlations, cell-weighted groups or absolute expression.
        sample_mean = np.mean([pearsonr((pred[i]-controls[('K562',batches[i])]).numpy(),
                                      (truth[i]-controls[('K562',batches[i])]).numpy()).statistic for i in range(4)])
        self.assertGreater(abs(score-sample_mean), .001)

    def test_controls_only_sparse_and_no_global_metric_cache(self):
        from gspp.metrics import CELL_TYPE_MEAN_CONTROL, PERT_MEANS
        before = (copy.deepcopy(CELL_TYPE_MEAN_CONTROL), copy.deepcopy(PERT_MEANS))
        obs = pd.DataFrame({'condition':['ctrl','ctrl','pert'], 'cell_line':['K562']*3,
                            'batch':['a','a','a']}, index=['0','1','2'])
        adata = anndata.AnnData(csr_matrix([[1.,2.,3.],[3.,4.,5.],[999.,999.,999.]]), obs=obs)
        means = control_means(adata)
        torch.testing.assert_close(means[('K562','a')], torch.tensor([2.,3.,4.],dtype=torch.float64))
        self.assertEqual(set(CELL_TYPE_MEAN_CONTROL), set(before[0]))
        self.assertEqual(set(PERT_MEANS), set(before[1]))
        self.assertIn(('K562',None), control_means(adata, match_batch=False))

    def test_missing_controls_and_nonfinite_predictions_fail_closed(self):
        batch = SimpleNamespace(pert_cond_names=['p'],pert_dose=['1'],cell_types=['K562'],experimental_batches=['a'])
        metric = ValidationDelta({})
        with self.assertRaises(ValueError): metric.update(torch.ones(1,3), torch.ones(1,3), batch)
        metric = ValidationDelta({('K562','a'): torch.zeros(3,dtype=torch.float64)})
        metric.update(torch.full((1,3),float('nan')), torch.ones(1,3), batch)
        self.assertTrue(np.isnan(metric.compute()[0]))


class EvidenceIntegrationTests(unittest.TestCase):
    def test_real_gat_hold_baseline_parity_and_evidence_resume(self):
        torch.set_num_threads(1)
        cfg = {**CONFIG, 'interval_steps':2, 'trend_window':2, 'trend_sample_every':1,
               'event_loss_ratio':1000., 'event_grad_ratio':1000.}
        final_states = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for enabled in (False,True):
                torch.manual_seed(42)
                model = model_fixture(); loader = loader_fixture()
                folder = root/str(enabled); folder.mkdir()
                client = Client(cfg)
                controller = ReactiveController(cfg, folder, PROMPT, enabled=enabled, client=client)
                trainer = Trainer(accelerator='cpu',devices=1,max_steps=12,max_epochs=3,logger=False,
                                  enable_checkpointing=False,enable_progress_bar=False,num_sanity_val_steps=0,
                                  callbacks=[controller],gradient_clip_val=1.)
                with patch.object(model,'general_validation_epoch',side_effect=AssertionError('test leak')):
                    trainer.fit(model,loader,loader)
                final_states.append(copy.deepcopy(model.state_dict()))
                self.assertIn('reactive_val_pearson_delta',trainer.callback_metrics)
                if enabled:
                    self.assertEqual(controller.counts['changed'],0)
                    self.assertEqual([v['validation']['step'] for v in client.observations],[4,8])
                    checkpoint = folder/'resume.ckpt'; trainer.save_checkpoint(checkpoint)
                    restored_dir = root/'resumed'; restored_dir.mkdir()
                    restored = ReactiveController(cfg,restored_dir,PROMPT,client=client)
                    trainer2 = Trainer(accelerator='cpu',devices=1,max_steps=20,max_epochs=5,logger=False,
                                       enable_checkpointing=False,enable_progress_bar=False,num_sanity_val_steps=0,
                                       callbacks=[restored],gradient_clip_val=1.)
                    trainer2.fit(model_fixture(),loader,loader,ckpt_path=checkpoint)
                    # Any regression wake-up is separately labeled; no repeated validation is sent.
                    steps=[v['validation']['step'] for v in client.observations]
                    self.assertEqual(len(steps),len(set(steps)))
                    self.assertGreater(restored.counts['attempts'],controller.counts['attempts'])
            self.assertTrue(all(torch.equal(final_states[0][k],final_states[1][k]) for k in final_states[0]))


if __name__ == '__main__':
    unittest.main()
