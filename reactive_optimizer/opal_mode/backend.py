"""Qwen backend for the pinned official OPAL API. Static checks are not a sandbox."""
import ast
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from opal.backends.base import Backend
from opal.misc.boundary import check as boundary_check
from reactive_optimizer.core import ChatClient, write_json

ALLOWED_FILES = {'analysis.py', 'updater.py'}
ALLOWED_IMPORTS = {'torch', 'math', 'copy', 'collections', 'typing', 're', 'statistics', 'functools'}
FORBIDDEN_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars',
                   'getattr', 'setattr', 'delattr', 'input', 'breakpoint'}


def validate_code(files, limit):
    if not isinstance(files, dict) or not files or not set(files) <= ALLOWED_FILES:
        raise ValueError('Only analysis.py and updater.py can be edited')
    for name, source in files.items():
        if not isinstance(source, str) or not source.strip() or len(source) > limit:
            raise ValueError('Invalid source size')
        tree = ast.parse(source, filename=name)
        required = 'build' if name == 'updater.py' else 'analyze'
        if not any(isinstance(node, ast.FunctionDef) and node.name == required for node in tree.body):
            raise ValueError(f'{name} requires {required}()')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.split('.')[0] not in ALLOWED_IMPORTS for alias in node.names):
                    raise ValueError('Disallowed import')
            if isinstance(node, ast.ImportFrom):
                if node.level or (node.module or '').split('.')[0] not in ALLOWED_IMPORTS:
                    raise ValueError('Disallowed import')
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                raise ValueError('Dynamic access or I/O is prohibited')
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                raise ValueError('Introspection is prohibited')
        findings = boundary_check({name: source})
        if findings:
            raise ValueError('OPAL boundary check flagged candidate')


def validate_options(cfg):
    for key in ('every', 'interval_steps', 'max_decisions', 'trial_steps', 'checkpoint_every',
                'max_tokens', 'max_source_chars', 'max_context_chars', 'preflight_steps',
                'preflight_max_numel', 'preflight_timeout_seconds'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['interval_steps'] % cfg['every']:
        raise ValueError('interval_steps must be a multiple of every')
    if type(cfg['blocking']) is not bool:
        raise ValueError('blocking must be boolean')
    for key in ('snapshot_limit_gb', 'preflight_memory_gb'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not isinstance(cfg['goal'], str) or not cfg['goal'].strip():
        raise ValueError('A goal is required')


def preflight(current, proposed, specs, options, record_dir):
    """Separate CPU process and timeout, not an OS security boundary."""
    job = {'current': current, 'proposed': proposed, 'specs': specs, 'options': options}
    worker = Path(__file__).with_name('preflight.py').resolve()
    with tempfile.TemporaryDirectory(prefix='txpert-opal-probe-') as tmp:
        root = Path(tmp)
        write_json(root / 'job.json', job)
        # No inherited authentication/proxy values are exposed to the child.
        env = {'PATH': os.defpath, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'TMPDIR': tmp}
        with (record_dir / 'preflight.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-I', str(worker), str(root / 'job.json')],
                                    cwd=tmp, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=options['preflight_timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise ValueError('Candidate preflight timed out') from None
            finally:
                # Clean up descendants if a candidate unexpectedly started any.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if code:
            raise ValueError('Candidate preflight failed; see preflight.log')


class QwenCodeBackend(Backend):
    name = 'qwen-local-code'

    def __init__(self, agent, options, specs, client=None):
        self.agent, self.options, self.specs = agent, options, specs
        self.instructions = Path(__file__).with_name('prompt.txt').read_text()
        self.client = client or ChatClient({**agent, 'max_tokens': options['max_tokens']}, self.instructions)
        self.history = []

    def state(self):
        return {'identity': {'agent': self.agent, 'options': self.options,
                             'instructions': self.instructions}, 'history': self.history}

    def restore(self, state):
        if state.get('identity') != self.state()['identity']:
            raise ValueError('Qwen code backend configuration/prompt changed during resume')
        self.history = state['history']

    def decide(self, workspace, message, system, record_dir):
        workspace, record_dir = Path(workspace), Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        current = {name: (workspace / name).read_text() for name in sorted(ALLOWED_FILES)}
        observation = {'opal_message': message, 'current_files': current, 'history': self.history,
                       'parameter_specs': self.specs}
        base_messages = [{'role': 'system', 'content': (system or '') + '\n\n' + self.instructions}]
        budget = self.options['max_context_chars'] - len(json.dumps(base_messages)) - len(json.dumps(observation))
        if budget < 0:
            return {'text': 'Context exceeds configured limit; retained current programs', 'returncode': 0}
        context = {}
        # Newest inbox first; additional bounded reference files. Never read arbitrary paths from model output.
        paths = sorted((workspace / 'inbox').glob('*.json'), reverse=True)
        paths += sorted((workspace / 'reference').glob('*.txt'))
        for path in paths:
            if path.is_symlink():
                continue
            if path.stat().st_size > budget:
                continue
            content = path.read_text()
            size = len(json.dumps({str(path.relative_to(workspace)): content}))
            if size <= budget:
                context[str(path.relative_to(workspace))] = content
                budget -= size
        observation['bounded_context'] = context
        payload = self.client.payload(observation)
        payload['messages'][0] = base_messages[0]
        write_json(record_dir / 'qwen-request.json', payload)
        write_json(record_dir / 'qwen-identity.json', {'model': self.agent['model'],
                   'revision': self.agent['revision'], 'precision': self.agent['precision']})
        result = {'status': 'kept', 'reason': 'No edit'}
        try:
            response = self.client.complete(payload)
            write_json(record_dir / 'qwen-response.json', response)
            proposal = json.loads(response['choices'][0]['message']['content'])
            if not isinstance(proposal, dict) or proposal.get('decision') not in ('keep', 'edit'):
                raise ValueError('Invalid decision schema')
            keys = {'decision', 'reason'} | ({'files'} if proposal['decision'] == 'edit' else set())
            if set(proposal) != keys or not isinstance(proposal['reason'], str) or not 1 <= len(proposal['reason']) <= 2000:
                raise ValueError('Invalid decision fields')
            if proposal['decision'] == 'edit':
                validate_code(proposal['files'], self.options['max_source_chars'])
                candidate = {**current, **proposal['files']}
                write_json(record_dir / 'candidate.json', candidate)
                preflight(current, candidate, self.specs, self.options, record_dir)
                # OPAL reads workspace only after decide returns. Restore both files if publication fails.
                try:
                    for name, source in proposal['files'].items():
                        target = workspace / name
                        if target.is_symlink():
                            raise ValueError('Program path must not be a symlink')
                        temporary = workspace / (name + '.candidate')
                        temporary.write_text(source)
                        temporary.replace(target)
                except Exception:
                    for name, source in current.items():
                        (workspace / name).write_text(source)
                    raise
                result['status'] = 'submitted_to_opal_trial'
            result['reason'] = proposal['reason']
        except Exception as exc:
            result = {'status': 'rejected', 'error_type': type(exc).__name__,
                      'reason': 'Proposal rejected; inspect response and preflight artifacts'}
            if isinstance(exc, (ValueError, SyntaxError)):
                result['diagnostic'] = str(exc)[:1000]
            log = record_dir / 'preflight.log'
            if log.exists():
                result['preflight_tail'] = log.read_text(errors='replace')[-4000:]
        write_json(record_dir / 'qwen-result.json', result)
        self.history = (self.history + [result])[-self.agent['history_size']:]
        return {'text': json.dumps(result), 'returncode': 0, 'cost_usd': None}
