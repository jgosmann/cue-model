from __future__ import absolute_import, print_function

import numpy as np

import nengo
from nengo.config import Default
from nengo.utils.compat import is_integer
from nengo.utils.least_squares_solvers import RandomizedSVD
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

from imem.amlearn import AML
from imem.modules import GatedMemory, SimilarityThreshold
from imem.utils.nengo import inhibit_net


class Control(nengo.Network):
    """Non-neural control signals for the TCM model.

    Parameters
    ----------
    protocol : FreeRecall
        Definition of experimental protocol.
    item_vocab : nengo_spa.Vocabulary
        Item vocabulary.

    Attributes
    ----------
    item_vocab : nengo_spa.Vocabulary
        Item vocabulary.
    output_pres_phase : nengo.Node
        Outputs 1 during the presentation phase, 0 otherwise.
    output_recall_phase : nengo.Node
        Outputs 1 during the recall phase, 0 otherwise.
    output_no_learn : nengo.Node
        Outputs 1 when learning should be disabled.
    output_stimulus : nengo.Node
        Outputs current stimulus.
    """
    def __init__(self, protocol, item_vocab):
        super(Control, self).__init__(label="Control")
        self.protocol = protocol
        self.item_vocab = item_vocab

        with self:
            self.output_pres_phase = nengo.Node(
                self.protocol.is_pres_phase, label='output_pres_phase')
            self.output_recall_phase = nengo.Node(
                self.protocol.is_recall_phase, label='output_recall_phase')

            self._current_stim = None
            self.output_no_learn = nengo.Node(
                lambda t: (self.protocol.is_recall_phase(t) or
                           self._current_stim is None or
                           self._current_stim.startswith('D')),
                label='output_no_learn')

            stimulus_fn = self.protocol.make_stimulus_fn()
            def store_current_stim(t):
                if self.protocol.is_pres_phase:
                    self._current_stim = stimulus_fn(t)
                    return self.item_vocab.parse(self._current_stim).v
                else:
                    return np.zeros(self.item_vocab.dimensions)
            self.output_stimulus = nengo.Node(
                store_current_stim, label='output_stimulus')


class TCM(spa.Network):
    """Network implementing the Temporal Context Model (TCM).

    Parameters
    ----------
    beta : float
        TCM beta parameter, the amount of context drift with each item.
    protocol : FreeRecall
        Experimental protocol.
    item_vocab : nengo_spa.Vocabulary or int
        Item vocabulary.
    context_vocab : nengo_spa.Vocabulary or int
        Context vocabulary.
    kwargs : dict
        Passed on to `nengo_spa.Network`.

    Attributes
    ----------
    item_vocab : nengo_spa.Vocabulary
        Item vocabulary.
    context_vocab : nengo_spa.Vocabulary
        Context vocabulary.
    output : nengo.Node
        Output of recalled vectors.
    """
    item_vocab = VocabularyOrDimParam(
        'item_vocab', optional=False, readonly=True)
    context_vocab = VocabularyOrDimParam(
        'context_vocab', optional=False, readonly=True)

    def __init__(
            self, beta, protocol, item_vocab=Default, context_vocab=Default,
            **kwargs):
        super(TCM, self).__init__(**kwargs)

        self.item_vocab = item_vocab
        if is_integer(context_vocab):
            context_vocab = spa.Vocabulary(context_vocab)
        self.context_vocab = context_vocab

        # Fill vocabularies
        self.item_vocab.populate(';'.join(protocol.get_all_items()))
        if protocol.n_distractors_per_epoch > 0:
            self.item_vocab.populate(';'.join(protocol.get_all_distractors()))
        for i in range(self.item_vocab.dimensions):
            self.context_vocab.populate('CTX' + str(i))

        with self:
            self.bias = nengo.Node(1.)
            self.ctrl = Control(protocol, self.item_vocab)

            # Association networks
            self.net_m_tf = AssocMatLearning(
                self.context_vocab, self.item_vocab)
            self.net_m_ft = AssocMatLearning(
                self.item_vocab, self.context_vocab,
                init_transform=self.context_vocab.vectors)

            # Stimulus input
            nengo.Connection(self.ctrl.output_stimulus, self.net_m_ft.input_cue)
            nengo.Connection(self.ctrl.output_stimulus, self.net_m_tf.input_target)

            # Context networks
            self.recalled_ctx = GatedMemory(self.context_vocab)
            nengo.Connection(self.net_m_ft.output, self.recalled_ctx.input)
            self.current_ctx = Context(beta, self.context_vocab)
            nengo.Connection(
                self.current_ctx.output, self.net_m_ft.input_target)

            nengo.Connection(self.recalled_ctx.output, self.current_ctx.input)

            # Determining update progress
            self.last_item = spa.State(self.item_vocab, feedback=1.)
            self.sim_th = SimilarityThreshold(self.item_vocab)
            nengo.Connection(self.ctrl.output_stimulus, self.sim_th.input_a)
            nengo.Connection(self.last_item.output, self.sim_th.input_b)
            nengo.Connection(self.bias, self.current_ctx.input_update_context)
            nengo.Connection(
                self.sim_th.output, self.current_ctx.input_update_context,
                transform=-1.)
            nengo.Connection(
                self.ctrl.output_stimulus, self.last_item.input, transform=1.,
                synapse=0.1)

            # Control of learning
            self.no_learn = nengo.Node(size_in=1)
            nengo.Connection(
                self.ctrl.output_no_learn, self.no_learn, transform=2.)
            nengo.Connection(self.no_learn, self.net_m_ft.no_learn)
            nengo.Connection(self.no_learn, self.net_m_tf.no_learn)

            nengo.Connection(self.bias, self.no_learn)
            nengo.Connection(self.sim_th.output, self.no_learn, transform=-1)

            # Recall
            self.recall = NeuralAccumulatorDecisionProcess(
                self.item_vocab.create_subset(protocol.get_all_items()))
            self.recall_gate = spa.State(self.item_vocab)
            nengo.Connection(self.current_ctx.output, self.net_m_tf.input_cue)
            nengo.Connection(self.net_m_tf.output, self.recall_gate.input)
            nengo.Connection(self.recall_gate.output, self.recall.input)
            nengo.Connection(
                self.recall.buf.mem.output, self.net_m_ft.input_cue)

            inhibit_net(self.ctrl.output_pres_phase, self.recall_gate)
            inhibit_net(
                self.ctrl.output_pres_phase, self.recall.buf.mem, strength=6)
            inhibit_net(self.ctrl.output_pres_phase, self.recall.state)
            inhibit_net(self.ctrl.output_pres_phase, self.recall.inhibit)

            nengo.Connection(self.recall.buf.output, self.sim_th.input_a)
            nengo.Connection(
                self.recall.buf.output, self.last_item.input, synapse=0.1)

            # Initialization of context
            initial_ctx = self.context_vocab.create_pointer().v
            ctx_init = nengo.Node(
                lambda t: initial_ctx if t < 0.3 else np.zeros(
                    self.context_vocab.dimensions))
            nengo.Connection(ctx_init, self.current_ctx.current.input)
            nengo.Connection(
                nengo.Node(lambda t: 4 if t < 0.3 else 0),
                self.current_ctx.old.input_store)

            self.output = self.recall.buf.output

        self.outputs = dict(default=(self.output, self.item_vocab))


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

            nengo.Connection(self.bias, self.current.input_store)
            nengo.Connection(
                self.input_update_context, self.current.input_store,
                transform=-1., synapse=None)

            nengo.Connection(self.input_update_context, self.old.input_store)

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
