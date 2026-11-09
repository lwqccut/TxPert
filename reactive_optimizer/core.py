"""Pure policy validation and HTTP transport; no generated code is executed."""
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_parameters(model):
    groups = {}
    prefixes = {'cntr_model': 'cell_encoder', 'pert_model': 'graph_encoder', 'decoder': 'decoder'}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = prefixes.get(name.split('.')[0], 'other')
            groups.setdefault(group, []).append(parameter)
    return [{'name': name, 'params': params} for name, params in groups.items()]


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric value')
    return float(value)


def validate_config(cfg):
    for key in ('interval_steps', 'history_size', 'max_tokens'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['feedback_enabled']) is not bool:
        raise ValueError('feedback_enabled must be boolean')
    for key in ('feedback_min_steps', 'feedback_baseline_max_age_steps',
                'feedback_patience', 'feedback_max_evaluations'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['feedback_cooldown_steps']) is not int or cfg['feedback_cooldown_steps'] < 0:
        raise ValueError('feedback_cooldown_steps must be nonnegative')
    if cfg['feedback_patience'] > cfg['feedback_max_evaluations']:
        raise ValueError('feedback_patience cannot exceed feedback_max_evaluations')
    for key in ('feedback_improvement', 'feedback_regression'):
        if not 0 < finite_number(cfg[key]) < 1:
            raise ValueError(f'{key} must be between 0 and 1')
    for low, high in [('lr_min', 'lr_max'), ('lr_factor_min', 'lr_factor_max'), ('clip_min', 'clip_max')]:
        if not 0 < finite_number(cfg[low]) <= finite_number(cfg[high]):
            raise ValueError(f'Invalid bounds: {low}, {high}')
    if not cfg['clip_min'] <= finite_number(cfg['clip_norm']) <= cfg['clip_max']:
        raise ValueError('clip_norm outside bounds')
    if finite_number(cfg['weight_decay_max']) < 0 or finite_number(cfg['timeout_seconds']) <= 0:
        raise ValueError('Invalid decay/timeout')


def validate_action(action, optimizer, cfg):
    """Validate the entire action before mutating any optimizer state."""
    if not isinstance(action, dict):
        raise ValueError('Action must be an object')
    kind = action.get('action')
    keys = {
        'hold': {'action', 'reason'},
        'scale_lr': {'action', 'group', 'factor', 'reason'},
        'set_weight_decay': {'action', 'group', 'value', 'reason'},
        'set_clip_norm': {'action', 'value', 'reason'},
    }
    if kind not in keys or set(action) != keys[kind]:
        raise ValueError('Unknown action or unexpected fields')
    if not isinstance(action['reason'], str) or not 1 <= len(action['reason']) <= 1000:
        raise ValueError('Expected a short reason')
    if kind == 'hold':
        return None, None, None
    if kind == 'set_clip_norm':
        value = finite_number(action['value'])
        if not cfg['clip_min'] <= value <= cfg['clip_max']:
            raise ValueError('Clip outside bounds')
        return None, 'clip_norm', value
    groups = {g['name']: g for g in optimizer.param_groups}
    if not isinstance(action['group'], str) or action['group'] not in groups:
        raise ValueError('Unknown parameter group')
    group = groups[action['group']]
    if kind == 'scale_lr':
        factor = finite_number(action['factor'])
        value = group['lr'] * factor
        if not cfg['lr_factor_min'] <= factor <= cfg['lr_factor_max'] or not cfg['lr_min'] <= value <= cfg['lr_max']:
            raise ValueError('Learning rate outside bounds')
        return group, 'lr', value
    value = finite_number(action['value'])
    if not 0 <= value <= cfg['weight_decay_max']:
        raise ValueError('Weight decay outside bounds')
    return group, 'weight_decay', value


class ChatClient:
    def __init__(self, cfg, prompt):
        self.cfg, self.prompt = cfg, prompt

    def payload(self, observation):
        return {
            'model': self.cfg['model'], 'temperature': self.cfg['temperature'],
            'seed': self.cfg['seed'], 'max_tokens': self.cfg['max_tokens'],
            'messages': [{'role': 'system', 'content': self.prompt},
                         {'role': 'user', 'content': json.dumps(observation, allow_nan=False)}],
            'response_format': {'type': 'json_object'},
            'chat_template_kwargs': {'enable_thinking': False},
        }

    def complete(self, payload):
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get(self.cfg['api_key_env'])
        if key:
            headers['Authorization'] = f'Bearer {key}'
        request = Request(self.cfg['endpoint'].rstrip('/') + '/chat/completions',
                          data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=self.cfg['timeout_seconds']) as response:
            return json.load(response)
