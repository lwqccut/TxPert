import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import opal
import torch
from opal import programs
from reactive_optimizer.core import ChatClient
from reactive_optimizer.opal_mode.backend import QwenCodeBackend, AgentSession
from reactive_optimizer.opal_mode.tests import test_opal as base
from reactive_optimizer.opal_mode.tests.fixtures import UPDATER, ANALYSIS


class ScriptedClient(ChatClient):
    def __init__(self, actions, workspace, original, protocol='json'):
        super().__init__(base.AGENT, base.PROMPT)
        self.actions, self.workspace, self.original, self.protocol = actions, workspace, original, protocol
        self.requests = []

    def complete(self, payload, timeout=None):
        # Until the finish tool returns, all live files must be unchanged.
        for name, source in self.original.items():
            if (self.workspace/name).read_text() != source:
                raise AssertionError('Candidate published before final submission')
        self.requests.append(copy.deepcopy(payload))
        action = self.actions[len(self.requests)-1]
        if self.protocol == 'openai':
            message = {'content': None, 'tool_calls': [{'id': f'call-{len(self.requests)}',
                       'type': 'function', 'function': {'name': action['tool'],
                        'arguments': json.dumps(action['arguments'])}}]}
        else:
            message = {'content': json.dumps(action)}
        return {'choices': [{'message': message}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def action(name, **arguments):
    return {'tool': name, 'arguments': arguments}


class AgentLoopTests(unittest.TestCase):
    def run_agent(self, actions, options=None, setup=None):
        _, _, specs, current = base.CodeTests().fixture()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        workspace = root/'workspace'; workspace.mkdir()
        programs.write_dir(workspace, current)
        if setup:
            setup(root, workspace)
        options = {**base.OPTIONS, 'max_agent_turns': len(actions), **(options or {})}
        client = ScriptedClient(actions, workspace, current, options['agent_protocol'])
        backend = QwenCodeBackend(base.AGENT, options, specs, client=client)
        result = backend.decide(workspace, 'training observations', 'official interface', root/'record')
        return root, client, json.loads(result['text']), current

    def test_read_diagnose_fail_repair_retest_submit_in_one_decision(self):
        bad = UPDATER.replace('self.optimizer.step()', 'self.optimizer.missing_method()')
        good = UPDATER.replace('import torch', 'import torch\nfrom helper_rate import RATE').replace('lr=0.002', 'lr=RATE')
        actions = [action('list_files'), action('read_file', path='updater.py'),
                   action('write_files', files={'updater.py':bad,'analysis.py':ANALYSIS}),
                   action('run_preflight'), action('read_file', path='checks/001/preflight.log'),
                   action('write_files', files={'updater.py':good,'helper_rate.py':'RATE = 0.002\n'}),
                   action('run_diagnostics', source=ANALYSIS), action('run_preflight'),
                   action('finish', decision='submit', reason='Fixed missing method; reset incompatible Adam state')]
        root, client, result, current = self.run_agent(actions)
        self.assertEqual(result['status'],'submitted_to_opal_trial')
        self.assertEqual(result['turns'],9)
        self.assertEqual(result['checks'],3)
        self.assertEqual(result['usage']['total_tokens'],135)
        self.assertIn('missing_method',json.dumps(client.requests[4]))
        self.assertIn('mean_gradient_norm',json.dumps(client.requests[7]))
        self.assertEqual((root/'workspace/updater.py').read_text(),good)
        self.assertTrue((root/'workspace/helper_rate.py').exists())
        self.assertEqual(len(list((root/'record/turns').glob('*/request.json'))),9)
        self.assertEqual(len(list((root/'record/checks').glob('*/candidate.json'))),3)

    def test_native_tool_calls_roundtrip(self):
        root, client, result, current = self.run_agent([
            action('read_file',path='analysis.py'), action('finish',decision='keep',reason='Read diagnostics first')],
            options={'agent_protocol':'openai'})
        self.assertEqual(result['status'],'kept')
        self.assertIn('tools',client.requests[0])
        self.assertNotIn('response_format',client.requests[0])
        self.assertEqual(client.requests[1]['messages'][-1]['role'],'tool')
        self.assertEqual(client.requests[1]['messages'][-1]['tool_call_id'],'call-1')

    def test_budget_exhaustion_does_not_publish_staged_code(self):
        root, client, result, current = self.run_agent([
            action('write_files',files={'updater.py':UPDATER})])
        self.assertEqual(result['status'],'budget_exhausted')
        self.assertEqual((root/'workspace/updater.py').read_text(),current['updater.py'])

    def test_edit_invalidates_passing_check(self):
        root, client, result, current = self.run_agent([
            action('write_files',files={'updater.py':UPDATER}), action('run_preflight'),
            action('write_files',files={'updater.py':UPDATER+'\n# modified after test\n'}),
            action('finish',decision='submit',reason='must fail stale check')])
        self.assertEqual(result['status'],'budget_exhausted')
        self.assertEqual((root/'workspace/updater.py').read_text(),current['updater.py'])
        last=json.loads((root/'record/turns/004/tool-result.json').read_text())
        self.assertIn('exact staged files',last['result']['message'])

    def test_file_tools_block_traversal_symlinks_and_unknown_paths(self):
        def setup(root, workspace):
            (root/'secret.txt').write_text('must not be read')
            (workspace/'reference').mkdir()
            (workspace/'reference/link.txt').symlink_to(root/'secret.txt')
        actions=[action('read_file',path='../secret.txt'),action('read_file',path='/etc/passwd'),
                 action('read_file',path='reference/link.txt'),
                 action('finish',decision='keep',reason='scope checked')]
        root, client, result, _ = self.run_agent(actions,setup=setup)
        self.assertEqual(result['status'],'kept')
        for request in client.requests:
            self.assertNotIn('must not be read',json.dumps(request))
        for turn in ('001','002','003'):
            self.assertEqual(json.loads((root/f'record/turns/{turn}/tool-result.json').read_text())['result']['status'],'error')

    def test_diagnostics_do_not_change_candidate_or_authorize_submission(self):
        root, client, result, current = self.run_agent([
            action('write_files',files={'updater.py':UPDATER}),
            action('run_diagnostics',source=ANALYSIS),
            action('finish',decision='submit',reason='diagnostic is not a candidate test')])
        self.assertEqual(result['status'],'budget_exhausted')
        self.assertEqual((root/'workspace/updater.py').read_text(),current['updater.py'])

    def test_syntax_error_returns_feedback_and_can_be_repaired(self):
        root, client, result, _ = self.run_agent([
            action('write_files',files={'updater.py':'def build('}),
            action('finish',decision='keep',reason='defer update')])
        self.assertIn('SyntaxError',json.dumps(client.requests[1]))
        self.assertEqual(result['status'],'kept')

    def test_cancelled_or_expired_checked_candidate_cannot_publish(self):
        _, _, specs, current = base.CodeTests().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root/'workspace'; workspace.mkdir()
            records = root/'records'; records.mkdir()
            programs.write_dir(workspace, current)
            session = AgentSession(workspace, records, current, specs, base.OPTIONS,
                                   time.monotonic()+60)
            session.call('write_files', {'files': {'updater.py': UPDATER}})
            with patch('reactive_optimizer.opal_mode.backend.preflight'):
                self.assertEqual(session.call('run_preflight', {})['status'], 'passed')
            session.cancelled = lambda: True
            with self.assertRaisesRegex(TimeoutError, 'cancelled'):
                session.publish()
            session.cancelled = lambda: False
            session.deadline = time.monotonic()-1
            with self.assertRaisesRegex(TimeoutError, 'budget'):
                session.publish()
            self.assertEqual((workspace/'updater.py').read_text(), current['updater.py'])

    def test_tool_budget_cannot_be_bypassed_by_finish(self):
        root, client, result, current = self.run_agent([
            action('write_files', files={'updater.py': UPDATER}),
            action('finish', decision='submit', reason='budget exhausted')],
            options={'max_tool_calls': 1})
        last = json.loads((root/'record/turns/002/tool-result.json').read_text())
        self.assertIn('Tool-call budget', last['result']['message'])
        self.assertEqual((root/'workspace/updater.py').read_text(), current['updater.py'])

    def test_multiturn_candidate_is_applied_by_official_wrapper(self):
        torch.set_num_threads(1)
        model, optimizer, specs, _ = base.CodeTests().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Official OPAL creates its own starting source; this client only scripts tool replies.
            client = ScriptedClient([
                action('read_file', path='analysis.py'),
                action('write_files', files={'updater.py': UPDATER, 'analysis.py': ANALYSIS}),
                action('run_preflight'),
                action('finish', decision='submit', reason='fixture SGD algorithm')], root, {})
            backend = QwenCodeBackend(base.AGENT, base.OPTIONS, specs, client=client)
            wrapped = opal.wrap(optimizer, model, total_steps=6, backend=backend,
                                goal='multiturn integration fixture', schedule=[2], every=1,
                                blocking=True, trial_steps=2, summarize=False, share_code=False,
                                run_dir=root/'run')
            try:
                for _ in range(6):
                    wrapped.zero_grad()
                    loss = model(torch.ones(2,2)).square().mean()
                    loss.backward()
                    wrapped.step(loss=loss)
                self.assertAlmostEqual(wrapped.param_groups[0]['lr'], 0.002)
                self.assertEqual(backend.history[-1]['status'], 'submitted_to_opal_trial')
                self.assertEqual(backend.history[-1]['turns'], 4)
                self.assertTrue(any(p.read_text() == UPDATER
                                    for p in (root/'run/programs').rglob('updater.py')))
            finally:
                wrapped.close()


if __name__=='__main__':
    unittest.main()
