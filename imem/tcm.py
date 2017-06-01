from __future__ import absolute_import, print_function

import numpy as np

import nengo
from nengo.config import Default
from nengo.utils.least_squares_solvers import RandomizedSVD
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

from imem.amlearn import AML
from imem.modules import GatedMemory, SimilarityThreshold
from imem.utils.nengo import inhibit_net


class TCMEquations(nengo.Network):
    """Provides direct implementation of TCM equations in nodes."""
    def __init__(self, control):
        super(TCMEquations, self).__init__(label="TCM Equations")

        self.control = control

        self._recall = np.zeros((1,))
        self.last_stim = None
        self.last_stim_change = 0

        with self:
            self.no_learn = nengo.Node(self._no_learn, size_in=1)
            self.recall = nengo.Node(self._set_recall, size_in=1)

    def _no_learn(self, t, x):
        if self.last_stim != self.control.current_stim:
            self.last_stim = self.control.current_stim
            self.last_stim_change = t

        if self.control.current_stim.startswith('D'):
            return True

        return not (
            x > .5 and self._recall[0] < 0.1 and True)
            # t - self.last_stim_change > 0.5)

    def _set_recall(self, t, x):
        self._recall[:] = x


class Control(nengo.Network):
    def __init__(self, protocol, item_d, distractor_rate, rng=np.random):
        super(Control, self).__init__(label="Control")
        self.protocol = protocol
        self.next_recall = 0

        self.vocab = spa.Vocabulary(item_d, rng=rng, strict=False)
        for i in range(self.protocol.n_items):
            self.vocab.add('V' + str(i), self.vocab.create_pointer())

        self.responses = []
        self.response_times = []

        with self:
            self.pres_phase = nengo.Node(self.protocol.is_pres_phase)
            self.recall_phase = nengo.Node(self.protocol.is_recall_phase)
            self.init_recall_ctx = nengo.Node(
                lambda t: len(self.responses) > 0)
            self.neg_init_recall_ctx = nengo.Node(
                lambda t: len(self.responses) <= 0 and
                self.protocol.is_recall_phase(t))
            self.response_in = nengo.Node(
                self._handle_response, size_in=item_d)
            self.recalled_out = nengo.Node(self._recalled_list)
            stimulus_fn = self.protocol.make_stimulus_fn(distractor_rate)
            self.current_stim = None
            def store_current_stim(t):
                if self.protocol.is_pres_phase:
                    self.current_stim = stimulus_fn(t)
                    return self.vocab.parse(self.current_stim).v
                else:
                    return np.zeros(item_d)
            self.stimulus = nengo.Node(store_current_stim)

    def _handle_response(self, t, x):
        dots = np.dot(self.vocab.vectors, x)
        idx = np.argmax(dots)
        if dots[idx] < 0.8:
            return dots[idx]
        if idx not in self.responses and (
                len(self.response_times) <= 0 or
                self.response_times[-1] + 0.1 < t):
            self.responses.append(idx)
            self.response_times.append(t)
        return dots[idx]

    def _recalled_list(self, t):
        if len(self.responses) <= 0:
            return np.zeros(self.vocab.dimensions)
        v = np.sum(self.vocab.vectors[self.responses], axis=0)
        v /= np.linalg.norm(v)
        return v


class TCM(spa.Network):
    item_vocab = VocabularyOrDimParam(
        'item_vocab', optional=False, readonly=True)
    context_vocab = VocabularyOrDimParam(
        'context_vocab', optional=False, readonly=True)

    def __init__(
            self, beta, control, item_vocab=Default,
            context_vocab=Default, **kwargs):
        super(TCM, self).__init__(**kwargs)

        self.item_vocab = item_vocab
        self.context_vocab = context_vocab

        with self:
            self.tcm = TCMEquations(control)
            self.ctrl = control

            # FIXME seed/generation of this
            v = spa.Vocabulary(
                self.context_vocab.dimensions, rng=np.random.RandomState(42))
            for i in range(self.item_vocab.dimensions):
                v.populate('CTX' + str(i))

            self.net_m_tf = AssocMatLearning(
                self.context_vocab, self.item_vocab)
            self.net_m_ft = AssocMatLearning(
                self.item_vocab, self.context_vocab,
                init_transform=v.vectors)

            nengo.Connection(self.ctrl.recall_phase, self.tcm.recall)

            nengo.Connection(self.ctrl.stimulus, self.net_m_ft.input_cue)
            nengo.Connection(self.tcm.no_learn, self.net_m_ft.no_learn)
            nengo.Connection(self.tcm.no_learn, self.net_m_tf.no_learn)

            self.recalled_ctx = GatedMemory(self.context_vocab)
            nengo.Connection(self.net_m_ft.output, self.recalled_ctx.input)

            self.current_ctx = Context(beta, self.context_vocab)
            nengo.Connection(
                self.current_ctx.output, self.net_m_ft.input_target)

            nengo.Connection(self.ctrl.stimulus, self.net_m_tf.input_target)

            nengo.Connection(self.recalled_ctx.output, self.current_ctx.input)

            self.last_item = spa.State(self.item_vocab, feedback=1.)
            self.sim_th = SimilarityThreshold(self.item_vocab)
            nengo.Connection(self.ctrl.stimulus, self.sim_th.input_a)
            nengo.Connection(self.last_item.output, self.sim_th.input_b)
            self.bias = nengo.Node(1.)
            nengo.Connection(self.bias, self.current_ctx.input_update_context)
            nengo.Connection(
                self.sim_th.output, self.current_ctx.input_update_context,
                transform=-1.)
            nengo.Connection(
                self.ctrl.stimulus, self.last_item.input, transform=1.,
                synapse=0.1)

            nengo.Connection(self.sim_th.output, self.tcm.no_learn)

            self.recall = NeuralAccumulatorDecisionProcess(self.ctrl.vocab)
            self.recall_gate = spa.State(self.item_vocab)
            nengo.Connection(self.current_ctx.output, self.net_m_tf.input_cue)
            nengo.Connection(self.net_m_tf.output, self.recall_gate.input)
            nengo.Connection(self.recall_gate.output, self.recall.input)
            nengo.Connection(
                self.recall.buf.mem.output, self.net_m_ft.input_cue)
            inhibit_net(self.ctrl.pres_phase, self.recall_gate)
            inhibit_net(self.ctrl.pres_phase, self.recall.buf.mem, strength=6)
            inhibit_net(self.ctrl.pres_phase, self.recall.state)
            inhibit_net(self.ctrl.pres_phase, self.recall.inhibit)

            nengo.Connection(self.recall.buf.output, self.sim_th.input_a)
            nengo.Connection(
                self.recall.buf.output, self.last_item.input, synapse=0.1)

            initial_ctx = self.context_vocab.create_pointer().v
            ctx_init = nengo.Node(
                lambda t: initial_ctx if t < 0.3 else np.zeros(
                    self.context_vocab.dimensions))
            nengo.Connection(ctx_init, self.current_ctx.current.input)
            nengo.Connection(
                nengo.Node(lambda t: 4 if t < 0.3 else 0),
                self.current_ctx.old.store)


class AssocMatLearning(spa.Network):
    input_vocab = VocabularyOrDimParam(
        'input_vocab', optional=False, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', optional=False, readonly=True)

    def __init__(
            self, input_vocab=Default, output_vocab=Default,
            init_transform=None, **kwargs):
        super(AssocMatLearning, self).__init__(**kwargs)

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        with self:
            d = self.input_vocab.dimensions
            self.state = spa.State(self.input_vocab, subdimensions=64)
            for e in self.state.all_ensembles:
                e.radius = 0.5
                e.intercepts = nengo.dists.Uniform(-1., 1.)
                e.eval_points = nengo.dists.UniformHypersphere(surface=False)
            self.target = spa.State(self.output_vocab)
            self.input_cue = self.state.input
            self.input_target = self.target.input
            self.output = nengo.Node(size_in=self.output_vocab.dimensions)

            for i, e in enumerate(self.state.all_ensembles):
                sd = e.dimensions
                start = i * sd
                end = (i + 1) * sd
                conn = nengo.Connection(
                    e, self.output[start:end],
                    learning_rule_type=AML(10.),
                    function=lambda x, sd=sd: np.zeros(sd),
                    solver=nengo.solvers.LstsqL2(solver=RandomizedSVD()))
                nengo.Connection(
                    self.target.output[start:end], conn.learning_rule)

            self.compare = SimilarityThreshold(self.output_vocab)
            nengo.Connection(self.output, self.compare.input_a)
            nengo.Connection(self.input_target, self.compare.input_b)
            inhibit_net(self.compare.output, self.target, strength=1.)

            if init_transform is not None:
                nengo.Connection(
                    self.state.output, self.output, transform=init_transform)

            self.no_learn = nengo.Node(size_in=1)
            inhibit_net(self.no_learn, self.target)

        self.inputs = {
            'default': (self.input_cue, self.input_vocab),
            'target': (self.input_target, self.output_vocab),
            'no_learn': (self.no_learn, None)}
        self.outputs = {'default': (self.output, self.output_vocab)}


class Context(spa.Network):
    """Network to store and update context in TCM fashion."""

    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    def __init__(self, beta, vocab=Default, **kwargs):
        super(Context, self).__init__(**kwargs)

        self.beta = beta
        self.vocab = vocab

        with self:
            self.input = nengo.Node(size_in=self.vocab.dimensions)

            self.new_ctx = nengo.Node(size_in=self.vocab.dimensions)
            self.current = GatedMemory(self.vocab)
            self.old = GatedMemory(self.vocab)

            nengo.Connection(self.input, self.new_ctx, transform=beta)
            nengo.Connection(
                self.old.output, self.new_ctx,
                transform=np.sqrt(1. - (beta)**2))
            nengo.Connection(self.new_ctx, self.current.input)
            nengo.Connection(self.current.output, self.old.input)

            self.bias = nengo.Node(1)
            self.input_update_context = nengo.Node(size_in=1)

            nengo.Connection(self.bias, self.current.store)
            nengo.Connection(
                self.input_update_context, self.current.store, transform=-1.,
                synapse=None)

            nengo.Connection(self.input_update_context, self.old.store)

        self.output = self.current.output
        self.inputs = dict(default=(self.input, vocab))
        self.outputs = dict(default=(self.output, vocab))


class NeuralAccumulatorDecisionProcess(spa.Network):
    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, noise=0., **kwargs):
        super(NeuralAccumulatorDecisionProcess, self).__init__(**kwargs)

        self.vocab = vocab

        n_items, d = self.vocab.vectors.shape

        with self:
            self.input = nengo.Node(size_in=d)

            with nengo.presets.ThresholdingEnsembles(0.):
                self.inp_thr = nengo.networks.EnsembleArray(50, n_items)

            with nengo.presets.ThresholdingEnsembles(0.):
                self.state = nengo.networks.EnsembleArray(50, n_items+1)
            nengo.Connection(
                self.input, self.inp_thr.input, transform=self.vocab.vectors,
                synapse=None)
            nengo.Connection(
                self.inp_thr.output, self.state.input[:-1], transform=0.1)
            nengo.Connection(self.state.output, self.state.input, synapse=0.1)

            with nengo.presets.ThresholdingEnsembles(0.8):
                self.threshold = nengo.networks.EnsembleArray(50, n_items+1)
            nengo.Connection(self.state.output, self.threshold.input)
            nengo.Connection(
                self.threshold.add_output(
                    'heaviside', lambda x: 1 if x > 0.8 else 0.),
                self.state.input,
                transform=-2 * (1. - np.eye(n_items+1)) + 1. * np.eye(
                    n_items + 1),
                synapse=0.1)

            self.buf = GatedMemory(self.vocab)
            self.inhibit = spa.State(self.vocab, feedback=1.)
            nengo.Connection(
                self.threshold.heaviside[:-1], self.buf.diff.input,
                transform=self.vocab.vectors.T)
            nengo.Connection(
                self.buf.mem.output, self.inhibit.input, transform=0.1)
            with nengo.presets.ThresholdingEnsembles(0.1):
                self.inhib_thr = nengo.networks.EnsembleArray(50, n_items)
            nengo.Connection(
                self.inhibit.output, self.inhib_thr.input,
                transform=self.vocab.vectors)
            nengo.Connection(
                self.inhib_thr.output, self.state.input[:-1], transform=-1.5)

            if noise > 0.:
                self.noise = nengo.Node(nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=0., std=0.06 * 0.2)),
                    size_out=n_items+1)
                nengo.Connection(self.noise, self.state.input)
