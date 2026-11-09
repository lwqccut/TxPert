import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from lightning import Trainer
from torch.utils.data import DataLoader

from reactive_optimizer.core import group_parameters, validate_action, validate_config, ChatClient
from reactive_optimizer.controller import ReactiveController

BASE = Path(__file__).resolve().parents[1]
CFG = {**yaml.safe_load((BASE / 'config.yaml').read_text()), 'feedback_enabled': False}
PROMPT = (BASE / 'prompt.txt').read_text()


class FakeClient(ChatClient):
    """Deterministic transport fixture, never claimed as an LLM experiment."""
    def complete(self, payload):
        return {'model': 'test-fixture', 'choices': [{'message': {'content': json.dumps({
            'action': 'scale_lr', 'group': 'graph_encoder', 'factor': 0.8, 'reason': 'fixture'})}}]}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.opt = torch.optim.AdamW([{'name': 'graph_encoder', 'params': [torch.nn.Parameter(torch.ones(2))]}], lr=0.001)

    def test_rejects_invalid_actions_without_mutating(self):
        original = copy.deepcopy(self.opt.state_dict())
        cases = [
            {'action': 'exec', 'code': 'bad', 'reason': 'x'},
            {'action': 'scale_lr', 'group': 'graph_encoder', 'factor': float('nan'), 'reason': 'x'},
            {'action': 'scale_lr', 'group': 'graph_encoder', 'factor': True, 'reason': 'x'},
            {'action': 'scale_lr', 'group': 'unknown', 'factor': 0.8, 'reason': 'x'},
            {'action': 'scale_lr', 'group': 'graph_encoder', 'factor': 10, 'reason': 'x'},
            {'action': 'set_clip_norm', 'value': -1, 'reason': 'x'},
            {'action': 'set_weight_decay', 'group': 'graph_encoder', 'value': 1, 'reason': 'x'},
            {'action': 'hold', 'reason': 'x', 'code': 'bad'},
        ]
        for action in cases:
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_action(action, self.opt, CFG)
        self.assertEqual(original, self.opt.state_dict())

    def test_allowed_actions_and_absolute_lr_bound(self):
        group, field, value = validate_action({'action':'scale_lr','group':'graph_encoder','factor':0.8,'reason':'x'}, self.opt, CFG)
        self.assertIs(group, self.opt.param_groups[0])
        self.assertEqual(field, 'lr')
        self.assertAlmostEqual(value, 0.0008)
        self.assertEqual(validate_action({'action':'hold','reason':'x'}, self.opt, CFG), (None,None,None))
        self.assertEqual(validate_action({'action':'set_clip_norm','value':2,'reason':'x'}, self.opt, CFG), (None,'clip_norm',2))
        self.opt.param_groups[0]['lr'] = CFG['lr_min']
        with self.assertRaises(ValueError):
            validate_action({'action':'scale_lr','group':'graph_encoder','factor':0.5,'reason':'x'},self.opt,CFG)

    def test_config_rejects_zero_interval(self):
        with self.assertRaises(ValueError):
            validate_config({**CFG, 'interval_steps': 0})


    def test_http_payload_and_response_without_exposing_key(self):
        client = ChatClient(CFG, PROMPT)
        payload = client.payload({'step': 500, 'validation': {'mse': 0.5}})
        response = {'model': CFG['model'], 'choices': [{'message': {'content':
                    '{"action":"hold","reason":"insufficient evidence"}'}}]}
        with patch.dict('os.environ', {'REACTIVE_LLM_API_KEY': 'test-only-secret'}):
            with patch('reactive_optimizer.core.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as request:
                self.assertEqual(client.complete(payload), response)
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, CFG['endpoint'] + '/chat/completions')
        self.assertEqual(json.loads(sent.data)['model'], 'Qwen/Qwen3-32B')
        self.assertFalse(json.loads(sent.data)['chat_template_kwargs']['enable_thinking'])
        self.assertNotIn('test-only-secret', json.dumps(payload))

    def test_resume_rejects_changed_prompt_or_policy(self):
        original = ReactiveController(CFG, '/tmp', PROMPT)
        changed = ReactiveController(CFG, '/tmp', PROMPT + 'changed')
        with self.assertRaises(ValueError):
            changed.load_state_dict(original.state_dict())

    def test_baseline_never_calls_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {**CFG, 'interval_steps': 1}
            controller = ReactiveController(cfg, tmp, PROMPT, enabled=False)
            trainer = SimpleNamespace(global_step=1, current_epoch=0, optimizers=[self.opt],
                                      save_checkpoint=lambda p: Path(p).touch())
            with patch.object(controller.client, 'complete', side_effect=AssertionError('unexpected API call')):
                controller.on_train_batch_end(trainer, None, {'loss': torch.tensor(1.)}, None, 0)
            self.assertEqual(self.opt.param_groups[0]['lr'], 0.001)
            self.assertEqual(controller.counts['attempts'], 0)


class TxPertIntegrationTests(unittest.TestCase):
    @staticmethod
    def model(scheduler=None):
        from reactive_optimizer.predictor import ReactivePredictor
        cfg = yaml.safe_load((BASE.parent / 'configs/config-gat.yaml').read_text())['model']
        graph = SimpleNamespace(graph_dict={'string': (torch.tensor([[0,1,2,3],[1,2,3,0]]).numpy(), torch.ones(4).numpy(), 4)})
        return ReactivePredictor(8, 8, 8, cfg, graph=graph, lr_scheduler_args=scheduler or {})

    @staticmethod
    def loaders():
        # Synthetic expression samples exercise real TxPert, not biological accuracy.
        def collate(rows):
            x = torch.stack(rows)
            from gspp.data.data_utils import Payload
            import pandas as pd
            return Payload(control=x, x=x * 0.5, p=torch.zeros(len(rows),1),
                           pert_idxs=pd.Series([[0] for _ in rows]),
                           pert_cond_names=pd.Series(['p0']*len(rows)),
                           pert_dose=pd.Series([1]*len(rows)), idx=pd.Series(range(len(rows))),
                           cell_types=pd.Series(['K562']*len(rows)),
                           experimental_batches=pd.Series(['b0']*len(rows)))
        return DataLoader([torch.randn(8) for _ in range(8)], batch_size=2, collate_fn=collate)

    def test_grouping_and_original_scheduler(self):
        model = self.model({'type':'cosine','total_epochs':4,'warmup_epochs':1})
        configured = model.configure_optimizers()
        opt = configured['optimizer']
        self.assertEqual([g['name'] for g in opt.param_groups], ['cell_encoder','graph_encoder','decoder'])
        params = [id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(params), len(set(params)))
        self.assertEqual(set(params), {id(p) for p in model.parameters() if p.requires_grad})
        for _ in range(4):
            opt.step()
            configured['lr_scheduler'].step()
        self.assertEqual(len(configured['lr_scheduler'].get_last_lr()), 3)

    def test_original_model_train_actions_checkpoint_and_resume(self):
        torch.set_num_threads(2)
        cfg = {**CFG, 'interval_steps': 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = ReactiveController(cfg, root, PROMPT, client=FakeClient(cfg, PROMPT))
            model = self.model()
            before = model.model.decoder[-1].weight.detach().clone()
            trainer = Trainer(accelerator='cpu', devices=1, max_steps=4, max_epochs=2,
                              logger=False, enable_checkpointing=False, enable_progress_bar=False,
                              num_sanity_val_steps=0, gradient_clip_val=1.0, callbacks=[controller])
            # The inherited method would inspect test data; our training must never call it.
            with patch.object(model, 'general_validation_epoch', side_effect=AssertionError('test leak')):
                trainer.fit(model, self.loaders(), self.loaders())
            self.assertFalse(torch.equal(before, model.model.decoder[-1].weight))
            events = sorted(root.glob('step-*'))
            self.assertEqual(len(events), 2)
            self.assertEqual(json.loads((events[0]/'decision.json').read_text())['status'], 'accepted')
            self.assertAlmostEqual(trainer.optimizers[0].param_groups[1]['lr'], 0.001*0.8**2)
            checkpoint = root / 'last.ckpt'
            trainer.save_checkpoint(checkpoint)
            state = torch.load(checkpoint, weights_only=False)
            self.assertTrue(state['optimizer_states'][0]['state'])
            self.assertIn('reactive_rng', state)
            self.assertIsNotNone(controller.val)
            self.assertNotIn('test', json.dumps(controller.history).lower())
            resume_dir = root/'resume'
            resume_dir.mkdir()
            resumed = ReactiveController(cfg, resume_dir, PROMPT, client=FakeClient(cfg,PROMPT))
            trainer2 = Trainer(accelerator='cpu', devices=1, max_steps=6, max_epochs=3,
                               logger=False, enable_checkpointing=False, enable_progress_bar=False,
                               num_sanity_val_steps=0, callbacks=[resumed])
            trainer2.fit(self.model(), self.loaders(), self.loaders(), ckpt_path=checkpoint)
            self.assertEqual(resumed.last_step, 6)
            self.assertEqual(len(list(resume_dir.glob('step-*'))), 1)
            self.assertAlmostEqual(trainer2.optimizers[0].param_groups[1]['lr'], 0.001*0.8**3)

    def test_feedback_trial_resumes_through_real_lightning_checkpoint(self):
        torch.set_num_threads(2)
        cfg = {**CFG, 'feedback_enabled': True, 'interval_steps': 2,
               'feedback_min_steps': 1, 'feedback_max_evaluations': 2,
               'feedback_cooldown_steps': 100}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = ReactiveController(cfg, root, PROMPT, client=FakeClient(cfg, PROMPT))
            trainer = Trainer(accelerator='cpu', devices=1, max_steps=6, max_epochs=3,
                              logger=False, enable_checkpointing=False, enable_progress_bar=False,
                              num_sanity_val_steps=0, callbacks=[controller])
            trainer.fit(self.model(), self.loaders(), self.loaders())
            self.assertIsNotNone(controller.trial)
            self.assertEqual(controller.trial['step'], 6)
            checkpoint = root / 'pending.ckpt'
            trainer.save_checkpoint(checkpoint)
            restored_dir = root / 'restored'
            restored_dir.mkdir()
            restored = ReactiveController(cfg, restored_dir, PROMPT, client=FakeClient(cfg, PROMPT))
            trainer2 = Trainer(accelerator='cpu', devices=1, max_steps=12, max_epochs=4,
                               logger=False, enable_checkpointing=False, enable_progress_bar=False,
                               num_sanity_val_steps=0, callbacks=[restored])
            trainer2.fit(self.model(), self.loaders(), self.loaders(), ckpt_path=checkpoint)
            self.assertEqual(restored.counts['attempts'], 1)
            self.assertIsNone(restored.trial)
            self.assertEqual(sum(restored.feedback_counts.values()), 1)
            self.assertEqual(len(restored.feedback_history), 2)
            saved = sorted((restored_dir / 'feedback').glob('*.ckpt'))[-1]
            state = torch.load(saved, weights_only=False)
            self.assertEqual(state['global_step'], 12)
            self.assertTrue(state['optimizer_states'][0]['state'])

    def test_failure_retains_optimizer_and_records_response(self):
        class BrokenClient(FakeClient):
            def complete(self, payload):
                return {'choices':[{'message':{'content':'```python\nraise Exception()'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            cfg={**CFG,'interval_steps':1}
            controller=ReactiveController(cfg,tmp,PROMPT,client=BrokenClient(cfg,PROMPT))
            opt=torch.optim.AdamW([{'name':'graph_encoder','params':[torch.nn.Parameter(torch.ones(2))]}],lr=0.001)
            trainer=SimpleNamespace(global_step=1,current_epoch=0,optimizers=[opt],save_checkpoint=lambda p: Path(p).touch())
            controller.on_train_batch_end(trainer,None,{'loss':torch.tensor(1.)},None,0)
            event=next(Path(tmp).glob('step-*'))
            self.assertEqual(json.loads((event/'decision.json').read_text())['status'],'rejected_or_unavailable')
            self.assertEqual(opt.param_groups[0]['lr'],0.001)
            self.assertTrue((event/'response.json').exists())


if __name__ == '__main__':
    unittest.main()
