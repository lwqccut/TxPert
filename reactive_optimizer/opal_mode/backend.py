"""Qwen backend for the pinned official OPAL API. Static checks are not a sandbox."""
import ast
import json
import math
import os
import re
import time
import copy
import hashlib
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from opal.backends.base import Backend
from opal.misc.boundary import check as boundary_check
from reactive_optimizer.core import ChatClient, write_json

ALLOWED_FILES = {'analysis.py', 'updater.py'}

def is_program(name):
    return isinstance(name, str) and (name in ALLOWED_FILES or re.fullmatch(r'helper_[A-Za-z0-9_]+\.py', name) is not None)
ALLOWED_IMPORTS = {'torch', 'math', 'copy', 'collections', 'typing', 're', 'statistics', 'functools'}
FORBIDDEN_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars',
                   'getattr', 'setattr', 'delattr', 'input', 'breakpoint'}


def validate_code(files, limit):
    if not isinstance(files, dict) or not files or not all(is_program(name) for name in files):
        raise ValueError('Only analysis.py, updater.py and root helper_*.py modules can be edited')
    for name, source in files.items():
        if not isinstance(source, str) or not source.strip() or len(source) > limit:
            raise ValueError('Invalid source size')
        tree = ast.parse(source, filename=name)
        required = {'updater.py': 'build', 'analysis.py': 'analyze'}.get(name)
        if required and not any(isinstance(node, ast.FunctionDef) and node.name == required for node in tree.body):
            raise ValueError(f'{name} requires {required}()')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.split('.')[0] not in ALLOWED_IMPORTS and not re.fullmatch(r'helper_[A-Za-z0-9_]+', alias.name) for alias in node.names):
                    raise ValueError('Disallowed import')
            if isinstance(node, ast.ImportFrom):
                if node.level or ((node.module or '').split('.')[0] not in ALLOWED_IMPORTS and not re.fullmatch(r'helper_[A-Za-z0-9_]+', node.module or '')):
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
    for key in ('max_agent_turns', 'max_tool_calls', 'decision_timeout_seconds', 'tool_result_chars'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['agent_protocol'] not in ('json', 'openai'):
        raise ValueError('agent_protocol must be json or openai')
    if cfg['interval_steps'] % cfg['every']:
        raise ValueError('interval_steps must be a multiple of every')
    if type(cfg['blocking']) is not bool:
        raise ValueError('blocking must be boolean')
    for key in ('snapshot_limit_gb', 'preflight_memory_gb'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not isinstance(cfg['goal'], str) or not cfg['goal'].strip():
        raise ValueError('A goal is required')
    if cfg.get('selection_metric', 'pearson_delta') not in ('pearson_delta', 'mse'):
        raise ValueError('selection_metric must be pearson_delta or mse')
    if type(cfg.get('share_code', False)) is not bool:
        raise ValueError('share_code must be boolean')


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


class AgentSession:
    """Per-decision staging, controlled tools, and transactional publication."""
    def __init__(self, workspace, record_dir, current, specs, options, deadline, cancelled=None):
        self.workspace = Path(workspace).resolve()
        self.records = Path(record_dir).resolve()
        self.current, self.staged = copy.deepcopy(current), copy.deepcopy(current)
        self.specs, self.options, self.deadline = specs, options, deadline
        self.checked_hash = None
        self.check_count = 0
        self.tool_calls = 0
        self.cancelled = cancelled or (lambda: False)

    def ensure_active(self):
        if self.cancelled():
            raise TimeoutError('Decision cancelled')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Decision time budget exhausted')

    def content_hash(self):
        return hashlib.sha256(json.dumps(self.staged, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def inside(root, path):
        if not isinstance(path, str) or '\\' in path:
            raise ValueError('Invalid path')
        rel = Path(path)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Path escapes allowed workspace')
        target = root / rel
        if not target.resolve().is_relative_to(root):
            raise ValueError('Symlink escapes allowed workspace')
        node = target
        while node != root:
            if node.is_symlink():
                raise ValueError('Symlink reads/writes are prohibited')
            node = node.parent
        return target

    def inventory(self):
        entries = [{'path': name, 'chars': len(source), 'staged': True}
                   for name, source in sorted(self.staged.items())]
        for folder in ('inbox', 'reference'):
            base = self.workspace / folder
            if base.is_symlink():
                continue
            for path in sorted(base.rglob('*')):
                if path.suffix not in ('.txt', '.json', '.py', '.md'):
                    continue
                rel = str(path.relative_to(self.workspace))
                try:
                    safe = self.inside(self.workspace, rel)
                    if safe.is_file():
                        entries.append({'path': rel, 'bytes': safe.stat().st_size, 'readonly': True})
                except ValueError:
                    continue
                if len(entries) >= 512:
                    return {'files': entries, 'truncated': True}
        for path in sorted((self.records/'checks').glob('*/preflight.log')):
            entries.append({'path': str(path.relative_to(self.records)), 'readonly': True})
        return {'files': entries, 'truncated': False}

    def read(self, arguments):
        path = arguments['path']
        start, count = arguments.get('start_line', 1), arguments.get('max_lines', 160)
        if type(start) is not int or start < 1 or type(count) is not int or not 1 <= count <= 400:
            raise ValueError('Invalid line range')
        if path in self.staged:
            content = self.staged[path]
        else:
            if not isinstance(path, str):
                raise ValueError('Path must be a string')
            parts = Path(path).parts
            if not parts:
                raise ValueError('Empty path')
            if parts[0] in ('inbox', 'reference'):
                target = self.inside(self.workspace, path)
            elif parts[0] == 'checks':
                target = self.inside(self.records, path)
            else:
                raise ValueError('Only programs, inbox, reference and checks are readable')
            if target.suffix not in ('.json', '.txt', '.md', '.py', '.log') or target.stat().st_size > 16*1024**2:
                raise ValueError('File type/size not permitted')
            content = target.read_text(errors='replace')
        lines = content.splitlines()
        selected, chars = [], 0
        for index in range(start-1, min(len(lines), start-1+count)):
            line = f'{index+1}: {lines[index]}'
            if selected and chars + len(line) > self.options['tool_result_chars']:
                break
            selected.append(line[:self.options['tool_result_chars']])
            chars += len(line)
        return {'path': path, 'content': '\n'.join(selected), 'total_lines': len(lines),
                'next_line': start + len(selected) if start + len(selected) <= len(lines) else None}

    def check(self, diagnostic=None):
        self.ensure_active()
        proposed = dict(self.staged)
        if diagnostic is not None:
            proposed['analysis.py'] = diagnostic
        else:
            self.checked_hash = None
        validate_code(proposed, self.options['max_source_chars'])
        self.check_count += 1
        target = self.records/'checks'/f'{self.check_count:03d}'
        target.mkdir(parents=True)
        write_json(target/'candidate.json', proposed)
        remaining = max(1, math.ceil(self.deadline-time.monotonic()))
        limits = {**self.options, 'preflight_timeout_seconds': min(remaining, self.options['preflight_timeout_seconds'])}
        status = 'passed'
        error = None
        try:
            preflight(self.current, proposed, self.specs, limits, target)
        except Exception as exc:
            status, error = 'failed', type(exc).__name__
        log = target/'preflight.log'
        tail = log.read_text(errors='replace')[-self.options['tool_result_chars']:] if log.exists() else ''
        # Compatibility artifact; full per-check logs remain separately available.
        if log.exists():
            (self.records/'preflight.log').write_text(log.read_text(errors='replace'))
        if status == 'passed' and diagnostic is None:
            self.checked_hash = self.content_hash()
        return {'status': status, 'error_type': error, 'log_tail': tail,
                'log_path': str(log.relative_to(self.records)),
                'candidate_hash': self.content_hash(), 'synthetic_cpu_probe': True}

    def publish(self):
        if self.checked_hash != self.content_hash():
            raise ValueError('Run preflight successfully on these exact staged files before submitting')
        self.ensure_active()
        write_json(self.records/'candidate.json', self.staged)
        # OPAL consumes the workspace after decide returns; no partial version is visible to training.
        targets = {name: self.inside(self.workspace, name) for name in self.staged}
        try:
            for name, target in targets.items():
                self.ensure_active()
                temporary = self.inside(self.workspace, name+'.candidate')
                temporary.write_text(self.staged[name])
                temporary.replace(target)
            self.ensure_active()
        except Exception:
            for name, target in targets.items():
                if name in self.current:
                    target.write_text(self.current[name])
                elif target.exists():
                    target.unlink()
            raise

    def call(self, name, arguments):
        self.tool_calls += 1
        if self.tool_calls > self.options['max_tool_calls']:
            raise ValueError('Tool-call budget exhausted')
        self.ensure_active()
        from .tools import TOOLS
        definition = next((t['function'] for t in TOOLS if t['function']['name'] == name), None)
        if definition is None or not isinstance(arguments, dict):
            raise ValueError('Unknown tool or invalid arguments')
        schema = definition['parameters']
        if not set(schema['required']) <= arguments.keys() or not arguments.keys() <= schema['properties'].keys():
            raise ValueError('Missing or unexpected tool arguments')
        if name == 'list_files':
            return self.inventory()
        if name == 'read_file':
            return self.read(arguments)
        if name == 'write_files':
            files = arguments['files']
            validate_code(files, self.options['max_source_chars'])
            candidate = {**self.staged, **files}
            if len(candidate) > 32 or sum(len(x) for x in candidate.values()) > self.options['max_context_chars']:
                raise ValueError('Candidate program bundle exceeds budget')
            self.staged = candidate
            self.checked_hash = None
            write_json(self.records/'staged.json', self.staged)
            return {'status': 'staged', 'files': sorted(files), 'candidate_hash': self.content_hash()}
        if name == 'run_preflight':
            return self.check()
        if name == 'run_diagnostics':
            return self.check(diagnostic=arguments['source'])
        reason, decision = arguments['reason'], arguments['decision']
        if not isinstance(reason, str) or not 1 <= len(reason) <= 2000 or decision not in ('submit', 'keep'):
            raise ValueError('Invalid final decision')
        if decision == 'submit' and self.staged != self.current:
            self.publish()
            return {'status': 'submitted_to_opal_trial', 'reason': reason, 'finished': True}
        return {'status': 'kept', 'reason': reason, 'finished': True}


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
            raise ValueError('Backend configuration/prompt changed during resume')
        self.history = state['history']

    def decide(self, workspace, message, system, record_dir):
        from .tools import TOOLS
        workspace, record_dir = Path(workspace), Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        current = {}
        for path in workspace.glob('*.py'):
            if is_program(path.name):
                safe = AgentSession.inside(workspace.resolve(), path.name)
                current[path.name] = safe.read_text()
        if not ALLOWED_FILES <= current.keys():
            raise ValueError('Missing OPAL program entrypoints')
        deadline = time.monotonic() + self.options['decision_timeout_seconds']
        session = AgentSession(workspace, record_dir, current, self.specs, self.options, deadline,
                               cancelled=lambda: getattr(self, 'cancelled', False))
        observation = {'opal_message': message, 'files': session.inventory(), 'history': self.history,
                       'parameter_specs': self.specs, 'current_files': current}
        protocol = self.options['agent_protocol']
        system_text = (system or '') + '\n\n' + self.instructions
        if protocol == 'json':
            system_text += '\nTool schemas: ' + json.dumps(TOOLS)
        header = [{'role': 'system', 'content': system_text},
                  {'role': 'user', 'content': json.dumps(observation)}]
        groups, transcript = [], []
        result = {'status': 'budget_exhausted', 'reason': 'No final decision; original programs retained'}
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        omitted, turns = 0, 0
        write_json(record_dir/'qwen-identity.json', {'model': self.agent['model'],
                   'revision': self.agent['revision'], 'precision': self.agent['precision']})
        for turn in range(1, self.options['max_agent_turns']+1):
            if getattr(self, 'cancelled', False) or time.monotonic() >= deadline:
                result = {'status': 'cancelled_or_timed_out', 'reason': 'Original programs retained'}
                break
            messages = header + [msg for group in groups for msg in group]
            while groups and len(json.dumps(messages)) > self.options['max_context_chars']:
                groups.pop(0)
                omitted += 1
                messages = header + [msg for group in groups for msg in group]
            if len(json.dumps(messages)) > self.options['max_context_chars']:
                result = {'status': 'context_exhausted', 'reason': 'Original programs retained'}
                break
            turns = turn
            folder = record_dir/'turns'/f'{turn:03d}'
            folder.mkdir(parents=True)
            payload = self.client.payload({})
            payload['messages'] = messages
            if protocol == 'openai':
                payload.pop('response_format', None)
                payload.update(tools=TOOLS, tool_choice='auto', parallel_tool_calls=False)
            write_json(folder/'request.json', payload)
            if turn == 1:
                write_json(record_dir/'qwen-request.json', payload)
            group = []
            finished = False
            try:
                response = self.client.complete(payload, timeout=min(self.agent['timeout_seconds'], max(0.1, deadline-time.monotonic())))
                write_json(folder/'response.json', response)
                write_json(record_dir/'qwen-response.json', response)
                for key in usage:
                    value = response.get('usage', {}).get(key, 0)
                    if type(value) is int and value >= 0:
                        usage[key] += value
                reply = response['choices'][0]['message']
                calls = reply.get('tool_calls') if protocol == 'openai' else None
                assistant = {k: reply[k] for k in ('content', 'tool_calls', 'reasoning_content') if k in reply}
                assistant['role'] = 'assistant'
                group.append(assistant)
                if calls:
                    if not isinstance(calls, list) or len(calls) != 1:
                        raise ValueError('Exactly one tool call per turn is required')
                    call = calls[0]
                    name = call['function']['name']
                    arguments = json.loads(call['function']['arguments'])
                    call_id = call['id']
                else:
                    action = json.loads(reply['content'])
                    if not isinstance(action, dict):
                        raise ValueError('Expected a JSON tool request')
                    # Legacy final proposal shorthand still enters the same validation loop.
                    if action.get('decision') in ('edit', 'keep'):
                        reason = action.get('reason')
                        if not isinstance(reason, str) or not 1 <= len(reason) <= 2000:
                            raise ValueError('Invalid rationale')
                        if action['decision'] == 'edit':
                            session.call('write_files', {'files': action['files']})
                            check = session.call('run_preflight', {})
                            if check['status'] != 'passed':
                                group.append({'role': 'user', 'content': json.dumps({'tool_result': check, 'instruction': 'Repair in this decision; staged changes are not published.'})})
                                groups.append(group)
                                transcript.extend(group)
                                write_json(folder/'tool-result.json', check)
                                continue
                        name, arguments = 'finish', {'decision': 'submit' if action['decision'] == 'edit' else 'keep', 'reason': reason}
                    else:
                        if set(action) != {'tool', 'arguments'}:
                            raise ValueError('Expected tool and arguments')
                        name, arguments = action['tool'], action['arguments']
                    call_id = None
                if getattr(self, 'cancelled', False):
                    raise TimeoutError('Decision cancelled')
                try:
                    tool_result = session.call(name, arguments)
                except Exception as exc:
                    tool_result = {'status': 'error', 'error_type': type(exc).__name__,
                                   'message': str(exc)[:1500] if isinstance(exc, (ValueError, SyntaxError, TimeoutError)) else 'Tool failed; correct arguments or inspect check artifacts'}
                write_json(folder/'tool-result.json', {'tool': name, 'arguments': arguments, 'result': tool_result})
                if call_id is not None:
                    group.append({'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(tool_result)})
                else:
                    group.append({'role': 'user', 'content': json.dumps({'tool': name, 'tool_result': tool_result})})
                if tool_result.get('finished'):
                    result = tool_result
                    finished = True
            except Exception as exc:
                # Do not print HTTP errors containing URLs/authentication material.
                error = {'status': 'error', 'error_type': type(exc).__name__,
                         'message': str(exc)[:1000] if isinstance(exc, (ValueError, SyntaxError)) else 'Request failed; retry within remaining budget'}
                # Never leave dangling native tool_calls in the next request.
                group = [{'role': 'user', 'content': json.dumps({'protocol_error': error})}]
                write_json(folder/'error.json', error)
            groups.append(group)
            transcript.extend(group)
            if finished:
                break
        result.update(turns=turns, tool_calls=session.tool_calls, checks=session.check_count,
                      usage=usage, omitted_context_groups=omitted)
        write_json(record_dir/'transcript.json', transcript)
        write_json(record_dir/'qwen-result.json', result)
        self.history = (self.history + [result])[-self.agent['history_size']:]
        return {'text': json.dumps(result), 'returncode': 0, 'cost_usd': None}
