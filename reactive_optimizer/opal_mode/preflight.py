"""Isolated-process CPU contract test, not a security sandbox or accuracy test."""
import copy
import inspect
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import torch
from opal import programs


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def finite_state(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite optimizer state')
    elif isinstance(value, dict):
        for x in value.values():
            finite_state(x)
    elif isinstance(value, (tuple, list)):
        for x in value:
            finite_state(x)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite scalar state')


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    options, specs = job['options'], job['specs']
    torch.set_num_threads(1)
    torch.manual_seed(0)
    memory = int(options['preflight_memory_gb'] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = options['preflight_timeout_seconds']
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**2, 8*1024**2))
    if sum(math.prod(p['shape']) for p in specs['parameters']) > options['preflight_max_numel']:
        raise ValueError('Parameter shapes exceed CPU preflight budget')
    named = [(p['name'], torch.nn.Parameter(torch.randn(p['shape']) * 0.01)) for p in specs['parameters']]
    by_name = dict(named)
    groups = [{**g, 'params': [by_name[name] for name in g['params']]} for g in specs['groups']]
    user = torch.optim.AdamW(groups)
    spec = {'total_updates': specs['total_steps'], 'update': 0, 'world_size': 1, 'rank': 0,
            'user_optimizers': [user], 'user_schedulers': [], 'clip_grad_norm': specs['clip_norm']}
    current = programs.load(job['current'], 'updater.py', 'previous', Path.cwd())
    previous = current.build(named, previous=None, spec=spec)

    def gradients():
        for _, p in named:
            p.grad = torch.randn_like(p) * 0.01
        return {name: p.grad for name, p in named}

    def step(updater, grads, number):
        with torch.no_grad():
            if len(inspect.signature(updater.step).parameters) == 3:
                updater.step(grads, number, SimpleNamespace(loss=torch.tensor(1.0)))
            else:
                updater.step(grads, number)

    step(previous, gradients(), 1)
    original = copy.deepcopy(user.state_dict())
    previous_state = copy.deepcopy(previous.state_dict())
    weights = {n: p.detach().clone() for n, p in named}
    module = programs.load(job['proposed'], 'updater.py', 'candidate', Path.cwd())
    candidate = module.build(named, previous=previous, spec={**spec, 'update': 1})
    if not equal(original, user.state_dict()) or not equal(previous_state, previous.state_dict()):
        raise ValueError('build mutated shared optimizer/previous state')
    if any(not torch.equal(weights[n], p) for n, p in named):
        raise ValueError('build mutated parameters')
    if sorted(id(p) for g in candidate.param_groups for p in g['params']) != sorted(id(p) for _, p in named):
        raise ValueError('param_groups must contain each parameter exactly once')
    analysis = programs.load(job['proposed'], 'analysis.py', 'analysis', Path.cwd())
    memory_state = {}
    reports = []
    for index in range(options['preflight_steps']):
        grads = gradients()
        weights = {n: p.detach().clone() for n, p in named}
        original_grads = {n: g.clone() for n, g in grads.items()}
        original_state = copy.deepcopy(candidate.state_dict())
        obs = SimpleNamespace(update=index+2, total_updates=specs['total_steps'],
                              losses=[1.0, 0.9], params=dict(named), grads=grads,
                              updater=candidate, metrics=[], program_ms_per_update=0.0, activations={})
        with torch.no_grad():
            report = analysis.analyze(obs, memory_state)
        json.dumps(report, allow_nan=False)
        reports.append({'probe_step': index+2, 'analysis': report})
        if (any(not torch.equal(weights[n], p) for n, p in named)
                or any(not torch.equal(original_grads[n], p.grad) for n, p in named)
                or not equal(original_state, candidate.state_dict())):
            raise ValueError('analysis mutated training tensors or optimizer state')
        step(candidate, grads, index+2)
        for name, param in named:
            if param.shape != weights[name].shape or param.dtype != torch.float32 or param.device.type != 'cpu':
                raise ValueError('Parameter shape, dtype or device changed')
            if not torch.isfinite(param).all():
                raise ValueError('Nonfinite update')
        state = candidate.state_dict()
        if programs.unpicklable(state) or programs.unpicklable(memory_state):
            raise ValueError('State cannot be safely checkpointed')
        finite_state(state)
        restored = copy.deepcopy(state)
        candidate.load_state_dict(restored)
        if not equal(state, candidate.state_dict()):
            raise ValueError('Optimizer state roundtrip failed')
    print(json.dumps({'passed': True, 'steps': options['preflight_steps'], 'reports': reports,
                      'scope': 'synthetic CPU contract/state checks; not proof of effectiveness or isolation'}))


if __name__ == '__main__':
    main(sys.argv[1])
