"""Deterministic coding fixtures; not Qwen-generated scientific results."""
ANALYSIS = '''import torch
EVERY = 1

def analyze(obs, memory):
    memory['calls'] = memory.get('calls', 0) + 1
    norms = [float(g.norm()) for g in obs.grads.values() if g is not None]
    return {'report': {'mean_gradient_norm': sum(norms) / max(len(norms), 1),
                       'calls': memory['calls']}}
'''
UPDATER = '''import torch

class Updater:
    follow_user_scheduler = False
    def __init__(self, named_params):
        self.optimizer = torch.optim.SGD([p for n, p in named_params], lr=0.002, momentum=0.9)
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    def step(self, grads, update):
        self.optimizer.step()
    def state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

def build(named_params, previous=None, spec=None):
    # Deliberate algorithm switch; Adam moments are not SGD momentum.
    return Updater(named_params)
'''
