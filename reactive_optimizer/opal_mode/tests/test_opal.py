import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
import opal
from lightning import Trainer
from opal import programs

from reactive_optimizer.core import ChatClient
from reactive_optimizer.opal_mode.backend import QwenCodeBackend, validate_code, preflight
from reactive_optimizer.opal_mode.integration import OpalPredictor, OpalAudit
from reactive_optimizer.opal_mode.tests.fixtures import UPDATER, ANALYSIS
from reactive_optimizer.tests import test_reactive as base_tests

ROOT = Path(__file__).resolve().parents[2]
AGENT = yaml.safe_load((ROOT/'config.yaml').read_text())
OPTIONS = yaml.safe_load((ROOT/'opal_mode/config.yaml').read_text())
PROMPT = (ROOT/'opal_mode/prompt.txt').read_text()


class FixtureClient(ChatClient):
    def complete(self, payload):
        return {'model':'fixture-not-a-real-llm', 'choices':[{'message':{'content':json.dumps({
            'decision':'edit','reason':'fixture: replace AdamW with SGD; reset incompatible state',
            'files':{'updater.py':UPDATER,'analysis.py':ANALYSIS}})}}]}


class CodeTests(unittest.TestCase):
    def fixture(self):
        model=torch.nn.Linear(2,2)
        opt=torch.optim.AdamW(model.parameters(),lr=0.001)
        specs={'parameters':[{'name':n,'shape':list(p.shape)} for n,p in model.named_parameters()],
               'groups':[{'params':[n for n,p in model.named_parameters()], 'lr':0.001,
                          'weight_decay':0.01,'betas':[0.9,0.999],'eps':1e-8}],
               'total_steps':10,'clip_norm':1.0}
        current=programs.starting_files([opt],[],{id(p):n for n,p in model.named_parameters()},every=1,clip=1.0)
        return model,opt,specs,current

    def test_code_scope_rejects_bad_paths_io_and_syntax(self):
        for files in [{'../evil.py':ANALYSIS}, {'analysis.py':'import os\ndef analyze(obs,memory):\n return os.environ'},
                      {'updater.py':'def build('}, {'analysis.py':'def analyze(obs,memory):\n return eval("1")'}]:
            with self.subTest(files=files), self.assertRaises((ValueError,SyntaxError)):
                validate_code(files,48000)
        validate_code({'analysis.py':ANALYSIS,'updater.py':UPDATER},48000)

    def test_backend_publishes_two_programs_only_after_cpu_preflight(self):
        _,_,specs,current=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workspace=root/'workspace'; workspace.mkdir()
            programs.write_dir(workspace,current)
            backend=QwenCodeBackend(AGENT,OPTIONS,specs,client=FixtureClient(AGENT,PROMPT))
            result=backend.decide(workspace,'fixture training report','OPAL interfaces',root/'record')
            self.assertEqual(json.loads(result['text'])['status'],'submitted_to_opal_trial')
            self.assertEqual((workspace/'updater.py').read_text(),UPDATER)
            self.assertTrue((root/'record/qwen-request.json').exists())
            self.assertIn('"passed": true',(root/'record/preflight.log').read_text())
            restored=QwenCodeBackend(AGENT,OPTIONS,specs)
            restored.restore(backend.state())
            self.assertEqual(restored.history,backend.history)

    def test_preflight_rejects_nonfinite_and_analysis_mutation(self):
        _,_,specs,current=self.fixture()
        for update,analysis in [(UPDATER.replace('self.optimizer.step()',
            "self.optimizer.step()\n        self.optimizer.param_groups[0]['params'][0].fill_(float('nan'))"),ANALYSIS),
            (UPDATER, "def analyze(obs, memory):\n    for p in obs.params.values():\n        p.zero_()\n    return {}\n")]:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                preflight(current,{'updater.py':update,'analysis.py':analysis},specs,OPTIONS,Path(tmp))

    def test_official_wrapper_applies_code_and_checkpoints_state(self):
        torch.set_num_threads(1)
        model,opt,_,_=self.fixture()
        def backend(workspace,message,system):
            (workspace/'updater.py').write_text(UPDATER)
            (workspace/'analysis.py').write_text(ANALYSIS)
            return 'fixture algorithm change'
        with tempfile.TemporaryDirectory() as tmp:
            wrapped=opal.wrap(opt,model,total_steps=8,backend=backend,goal='fixture MSE',
                              schedule=[2],every=1,blocking=True,trial_steps=2,
                              summarize=False,share_code=False,run_dir=Path(tmp)/'run')
            for _ in range(8):
                wrapped.zero_grad()
                loss=model(torch.ones(2,2)).square().mean()
                loss.backward(); wrapped.step(loss=loss)
            state=copy.deepcopy(wrapped.state_dict())
            wrapped.close()
            sources=list((Path(tmp)/'run/programs').rglob('updater.py'))
            self.assertTrue(any(p.read_text()==UPDATER for p in sources))
            model2=torch.nn.Linear(2,2); model2.load_state_dict(model.state_dict())
            opt2=torch.optim.AdamW(model2.parameters(),lr=0.001)
            resumed=opal.wrap(opt2,model2,total_steps=10,backend='keep',goal='fixture MSE',
                              schedule=[],every=1,summarize=False,share_code=False,run_dir=Path(tmp)/'resume')
            resumed.load_state_dict(state)
            resumed.zero_grad(); loss=model2(torch.ones(2,2)).square().mean()
            loss.backward(); resumed.step(loss=loss)
            self.assertEqual(resumed.update,9)
            self.assertAlmostEqual(resumed.param_groups[0]['lr'],0.002)
            resumed.close()

    def test_official_first_trial_failure_restores_parameters(self):
        torch.set_num_threads(1)
        def broken(workspace, message, system):
            source = UPDATER.replace('self.optimizer.step()',
                       "self.optimizer.param_groups[0]['params'][0].add_(100)\n        raise RuntimeError('fixture failure')")
            (workspace/'updater.py').write_text(source)
            return 'exercise official first-step rollback'
        results=[]
        with tempfile.TemporaryDirectory() as tmp:
            for use_opal in (False,True):
                torch.manual_seed(5)
                model=torch.nn.Linear(2,2)
                opt=torch.optim.AdamW(model.parameters(),lr=0.001)
                if use_opal:
                    opt=opal.wrap(opt,model,total_steps=5,goal='rollback fixture',backend=broken,
                                  schedule=[2],every=1,trial_steps=2,blocking=True,summarize=False,
                                  share_code=False,run_dir=Path(tmp)/'trial',max_decisions=1)
                for _ in range(5):
                    opt.zero_grad(); loss=model(torch.ones(2,2)).square().mean(); loss.backward()
                    if use_opal:
                        opt.step(loss=loss)
                    else:
                        opt.step()
                results.append(copy.deepcopy(model.state_dict()))
                if use_opal:
                    opt.close()
            self.assertTrue(all(torch.equal(results[0][k],results[1][k]) for k in results[0]))

    def test_asynchronous_decision_does_not_block_training(self):
        import threading
        import time
        started, release = threading.Event(), threading.Event()
        def backend(workspace, message, system):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError('Training did not progress while agent was waiting')
            (workspace/'updater.py').write_text(UPDATER)
            return 'asynchronous fixture'
        with tempfile.TemporaryDirectory() as tmp:
            model=torch.nn.Linear(2,2)
            wrapped=opal.wrap(torch.optim.AdamW(model.parameters(),lr=0.001),model,
                              total_steps=30,goal='async fixture',backend=backend,
                              schedule=[2],every=1,blocking=False,trial_steps=2,
                              share_code=False,summarize=False,run_dir=Path(tmp)/'run')
            progressed=False
            for step in range(30):
                wrapped.zero_grad(); loss=model(torch.ones(2,2)).square().mean(); loss.backward()
                wrapped.step(loss=loss)
                if started.is_set() and step >= 4:
                    progressed=True
                    release.set()
                time.sleep(0.005)
            wrapped.close()
            self.assertTrue(progressed)
            self.assertAlmostEqual(wrapped.param_groups[0]['lr'],0.002)

    def test_lightning_keep_bridge_runs_real_txpert_and_resumes(self):
        torch.set_num_threads(2)
        options={**OPTIONS,'every':1,'interval_steps':2,'checkpoint_every':2}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def model(directory):
                # Same constructor and architecture as phase-one real-model integration tests.
                base=base_tests.TxPertIntegrationTests.model()
                base.__class__=OpalPredictor
                base.configure_opal(AGENT,options,directory,keep=True)
                return base
            trainer=Trainer(accelerator='cpu',devices=1,max_steps=4,max_epochs=2,logger=False,
                            enable_progress_bar=False,enable_checkpointing=False,num_sanity_val_steps=0,
                            callbacks=[OpalAudit(AGENT,options,root,PROMPT,keep=True)])
            loader=base_tests.TxPertIntegrationTests.loaders()
            trainer.fit(model(root),loader,loader)
            trainer.save_checkpoint(root/'last.ckpt')
            target=root/'resume'; target.mkdir()
            trainer2=Trainer(accelerator='cpu',devices=1,max_steps=6,max_epochs=3,logger=False,
                             enable_progress_bar=False,enable_checkpointing=False,num_sanity_val_steps=0,
                             callbacks=[OpalAudit(AGENT,options,target,PROMPT,keep=True)])
            trainer2.fit(model(target),loader,loader,ckpt_path=root/'last.ckpt')
            self.assertEqual(trainer2.global_step,6)
            self.assertEqual(trainer2.optimizers[0].update,6)
            self.assertTrue((target/'opal-checkpoints/step-000000006.ckpt').exists())


if __name__=='__main__':
    unittest.main()
