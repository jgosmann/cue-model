import nengo
import numpy as np
from nengo.builder.connection import get_eval_points, solve_for_decoders
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.config import SupportDefaultsMixin
from nengo.params import Default, IntParam


class AML(nengo.learning_rules.LearningRuleType, SupportDefaultsMixin):
    error_type = 'decoded'
    modifies = 'decoders'
    seed = IntParam('seed', default=None, optional=True, readonly=True)

    def __init__(self, learning_rate=1., seed=Default):
        super(AML, self).__init__(learning_rate)
        self.seed = seed


class SimAML(nengo.builder.Operator):
    def __init__(self, learning_rate, base_decoders, pre, error, decoders,
                 tag=None):
        super(SimAML, self).__init__(tag=tag)

        self.learning_rate = learning_rate
        self.base_decoders = base_decoders

        self.sets = []
        self.incs = []
        self.reads = [pre, error]
        self.updates = [decoders]

    def make_step(self, signals, dt, rng):
        base_decoders = self.base_decoders
        pre = signals[self.pre]
        error = signals[self.error]
        decoders = signals[self.decoders]
        alpha = self.learning_rate * dt

        def step_assoc_learning():
            scale = 1.
            decoders[...] += alpha * scale * error[:, None] * np.dot(
                pre, base_decoders.T)

        return step_assoc_learning

    @property
    def pre(self):
        return self.reads[0]

    @property
    def error(self):
        return self.reads[1]

    @property
    def decoders(self):
        return self.updates[0]


@nengo.builder.Builder.register(AML)
def build_aml(model, aml, rule):
    if aml.seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(aml.seed)

    conn = rule.connection

    error = Signal(np.zeros(rule.size_in), name="aml:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error

    pre = model.sig[conn.pre_obj]['in']
    decoders = model.sig[conn]['weights']

    # TODO caching
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = eval_points

    x = np.dot(eval_points, encoders.T)

    base_decoders, _ = solve_for_decoders(
        conn.solver, conn.pre_obj.neuron_type, gain, bias, x, targets, rng=rng)

    model.add_op(SimAML(
        aml.learning_rate, base_decoders, pre, error, decoders))
