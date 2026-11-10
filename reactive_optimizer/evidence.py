"""Serializable bounded-agent evidence windows and decision scheduling."""
import copy
import math
import statistics


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(rows, key):
    points = [(row['step'], row[key]) for row in rows if finite(row.get(key))]
    if not points:
        return None
    steps, values = zip(*points)
    mean, spread = statistics.mean(values), statistics.pstdev(values)
    center = statistics.mean(steps)
    variance = sum((step-center)**2 for step in steps)
    slope = sum((step-center)*(value-mean) for step, value in points) / variance if variance else 0.0
    return {'count': len(values), 'start_step': steps[0], 'end_step': steps[-1],
            'mean': mean, 'std': spread, 'slope_per_step': slope,
            'relative_std': spread/max(abs(mean), 1e-12)}


class Evidence:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = []
        self.validations = []
        self.last_observed_step = -1
        self.validation_id = 0
        self.used_validation_id = 0
        self.next_validation_id = 1
        self.hold_streak = 0
        self.event_active = False
        self.event_id = 0
        self.used_event_id = 0
        self.event = None
        self.skipped = {}

    def state(self):
        return copy.deepcopy({key: value for key, value in vars(self).items() if key != 'cfg'})

    def restore(self, state):
        for key in self.state():
            setattr(self, key, copy.deepcopy(state[key]))

    def observe(self, step, loss, gradients):
        if step <= self.last_observed_step:
            return
        self.last_observed_step = step
        if self.samples and step-self.samples[-1]['step'] < self.cfg['trend_sample_every']:
            return
        row = {'step': step, 'loss': loss if finite(loss) else None}
        for name, values in gradients.items():
            grad, weight = values.get('grad_norm'), values.get('weight_norm')
            row[f'grad:{name}'] = grad
            row[f'grad_weight_ratio:{name}'] = grad/max(weight, 1e-12) if finite(grad) and finite(weight) else None
        self.samples.append(row)
        self.samples = self.samples[-2*self.cfg['trend_window']:]
        signals = self._signals()
        active = bool(signals)
        if active and not self.event_active:
            self.event_id += 1
            self.event = {'id': self.event_id, 'step': step, 'signals': signals}
        self.event_active = active

    def _signals(self):
        n = self.cfg['trend_window']
        if len(self.samples) < 2*n:
            return []
        previous, recent = self.samples[:n], self.samples[n:]
        signals = []
        keys = {'loss'} | {key for row in recent for key in row if key.startswith('grad:')}
        for key in sorted(keys):
            old, new = stats(previous, key), stats(recent, key)
            if old is None or new is None or old['count'] != n or new['count'] != n:
                continue
            factor = self.cfg['event_loss_ratio'] if key == 'loss' else self.cfg['event_grad_ratio']
            threshold = max(abs(old['mean']), 1e-12) * factor
            fraction = sum(row[key] > threshold for row in recent) / n
            if fraction >= self.cfg['event_fraction']:
                signals.append({'quantity': key, 'previous_mean': old['mean'],
                                'recent_mean': new['mean'], 'fraction_above_threshold': fraction})
        return signals

    def validate(self, value):
        if self.validations and value['step'] <= self.validations[-1]['step']:
            return False
        self.validation_id += 1
        self.validations.append(copy.deepcopy(value))
        self.validations = self.validations[-self.cfg['history_size']:]
        return True

    def trigger(self):
        if self.event_active and self.event_id > self.used_event_id:
            return {'kind': 'sustained_training_event', 'event': self.event}
        if self.validation_id <= self.used_validation_id:
            return None
        # A new deterioration is evidence even during hold backoff.
        if len(self.validations) >= 2:
            old, new = self.validations[-2:]
            mse_bad = (finite(old.get('mse')) and finite(new.get('mse')) and
                       new['mse'] > old['mse']*(1+self.cfg['feedback_regression']))
            pearson_bad = (finite(old.get('pearson_delta')) and finite(new.get('pearson_delta')) and
                           new['pearson_delta'] < old['pearson_delta']-self.cfg['feedback_pearson_regression'])
            if mse_bad or pearson_bad:
                return {'kind': 'validation_regression', 'mse_regressed': mse_bad,
                        'pearson_regressed': pearson_bad}
        if self.validation_id >= self.next_validation_id:
            return {'kind': 'fresh_validation', 'validation_id': self.validation_id}
        return None

    def consume(self):
        self.used_validation_id = self.validation_id
        if self.event_active:
            self.used_event_id = self.event_id

    def outcome(self, unchanged):
        self.hold_streak = self.hold_streak+1 if unchanged else 0
        exponent = min(30, max(0, self.hold_streak-self.cfg['hold_backoff_after']+1))
        spacing = min(2**exponent, self.cfg['hold_backoff_max_validations'])
        self.next_validation_id = self.validation_id+spacing

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0)+1

    def report(self):
        n = self.cfg['trend_window']
        recent = self.samples[-n:]
        previous = self.samples[-2*n:-n]
        keys = sorted({key for row in self.samples for key in row if key != 'step'})
        windows = {key: {'recent': stats(recent, key), 'previous': stats(previous, key)} for key in keys}
        return {'training_windows': windows, 'validation_history': self.validations,
                'validation_mse_trend': stats(self.validations, 'mse'),
                'validation_pearson_delta_trend': stats(self.validations, 'pearson_delta'),
                'hold_streak': self.hold_streak,
                'validations_until_next_routine_call': max(0, self.next_validation_id-self.validation_id),
                'note': 'Gradient/weight norm ratio is a scale proxy, not the actual AdamW update ratio.'}
