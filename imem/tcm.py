from __future__ import absolute_import, print_function

import nengo
from nengo.config import Default
from nengo.utils.least_squares_solvers import RandomizedSVD
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam
import numpy as np

from imem.amlearn import AML
from imem.modules import GatedMemory, SimilarityThreshold
from imem.utils.nengo import inhibit_net


class TCM(spa.Network):
    """Network implementing the Temporal Context Model (TCM).

    Parameters
    ----------
    beta : float
        TCM beta parameter, the amount of context drift with each item.
    protocol : Recall
        Experimental protocol.
    recall_noise : float
        Standard deviation of Gaussian noise to add in recall.
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

    # pylint: disable=too-many-statements,too-many-arguments
    def __init__(
            self, task_vocabs, protocol, ctrl, beta, recall_noise=0.,
            **kwargs):
        kwargs.setdefault('label', 'TCM')
        super(TCM, self).__init__(**kwargs)

        self.task_vocabs = task_vocabs

        with self:
            self.bias = nengo.Node(1.)
            self.ctrl = ctrl

            # Association networks
            self.net_m_tf = AssocMatLearning(
                self.task_vocabs.contexts, self.task_vocabs.items)
            self.net_m_ft = AssocMatLearning(
                self.task_vocabs.items, self.task_vocabs.contexts,
                init_transform=self.task_vocabs.contexts.vectors)

            # Stimulus input
            self.input = nengo.Node(size_in=self.task_vocabs.items.dimensions)
            nengo.Connection(self.input, self.net_m_ft.input_cue)
            nengo.Connection(self.input, self.net_m_tf.input_target)

            # Position input
            self.input_pos = nengo.Node(
                size_in=self.task_vocabs.positions.dimensions)
            nengo.Connection(self.input_pos, self.net_m_ft.input_cue)
            nengo.Connection(
                self.input_pos, self.net_m_tf.input_target)

            # Context networks
            self.current_ctx = Context(beta, self.task_vocabs.contexts)
            nengo.Connection(
                self.current_ctx.output, self.net_m_ft.input_target)
            nengo.Connection(self.net_m_ft.output, self.current_ctx.input)

            # Determining update progress
            self.last_item = spa.State(self.task_vocabs.items, feedback=1.)
            self.sim_th = SimilarityThreshold(self.task_vocabs.items)
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
            nengo.Connection(self.no_learn, self.net_m_ft.input_no_learn)
            nengo.Connection(self.no_learn, self.net_m_tf.input_no_learn)

            nengo.Connection(self.bias, self.no_learn)
            nengo.Connection(self.sim_th.output, self.no_learn, transform=-1)

            # Recall
            self.recall = NeuralAccumulatorDecisionProcess(
                self.task_vocabs.items.create_subset(protocol.get_all_items()),
                noise=recall_noise, min_evidence=.025, n_inputs=2)
            self.recall_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(self.current_ctx.output, self.net_m_tf.input_cue)
            nengo.Connection(self.net_m_tf.output, self.recall_gate.input)
            nengo.Connection(self.recall_gate.output, self.recall.input[0])

            self.recalled_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(
                self.recall.buf.mem.output, self.recalled_gate.input)
            nengo.Connection(
                self.recalled_gate.output, self.net_m_ft.input_cue)
            inhibit_net(
                self.ctrl.output_serial_recall, self.recalled_gate, strength=6)

            inhibit_net(self.ctrl.output_pres_phase, self.recall_gate)
            inhibit_net(
                self.ctrl.output_pres_phase, self.recall.buf.mem, strength=6)
            inhibit_net(self.ctrl.output_pres_phase, self.recall.state)
            inhibit_net(self.ctrl.output_pres_phase, self.recall.inhibit)

            nengo.Connection(self.recall.buf.output, self.sim_th.input_a)
            nengo.Connection(
                self.recall.buf.output, self.last_item.input, transform=0.1,
                synapse=0.1)

            if len(self.task_vocabs.positions) > 0:
                self.pos_recall = NeuralAccumulatorDecisionProcess(
                    self.task_vocabs.positions, noise=recall_noise)
                nengo.Connection(
                    self.recall_gate.output, self.pos_recall.input[0])
                nengo.Connection(
                    self.pos_recall.buf.mem.output, self.net_m_ft.input_cue)
                inhibit_net(
                    self.ctrl.output_serial_recall, self.pos_recall.buf)

                inhibit_net(
                    self.ctrl.output_pres_phase, self.pos_recall.buf.mem,
                    strength=6)
                inhibit_net(self.ctrl.output_pres_phase, self.pos_recall.state)
                inhibit_net(
                    self.ctrl.output_pres_phase, self.pos_recall.inhibit)

                nengo.Connection(
                    self.pos_recall.buf.output, self.sim_th.input_a)
                nengo.Connection(
                    self.pos_recall.buf.output, self.last_item.input,
                    transform=0.1, synapse=0.1)

            # Initialization of context
            initial_ctx = self.task_vocabs.contexts.create_pointer().v
            ctx_init = nengo.Node(
                lambda t: initial_ctx if t < 0.3 else np.zeros(
                    self.task_vocabs.contexts.dimensions))
            nengo.Connection(ctx_init, self.current_ctx.current.input)
            nengo.Connection(
                nengo.Node(lambda t: 4 if t < 0.3 else 0),
                self.current_ctx.old.input_store)

            self.output = self.recall.output
            self.output_stim_update_done = self.sim_th.output

        self.inputs = dict(
            default=(self.input, self.task_vocabs.items),
            input_pos=(self.input_pos, self.task_vocabs.positions))
        self.outputs = dict(
            default=(self.output, self.task_vocabs.items),
            stim_update_done=(self.output_stim_update_done, None))


class AssocMatLearning(spa.Network):
    """Association matrix learning network.

    Will stop learning for a cue-target pair once the cue recalls the target
    with similarity 1.

    Parameters
    ----------
    input_vocab : nengo_spa.Vocabulary or int
        Input vocabulary.
    output_vocab : nengo_spa.Vocabulary or int
        Output vocabulary.
    init_transform : ndarray
        Initial transform from input to output (before learning).
    kwargs : dict
        Passed on to `nengo_spa.Network`.

    Attributes
    ----------
    input_vocab : nengo_spa.Vocabulary
        Input vocabulary.
    output_vocab : nengo_spa.Vocabulary
        Output vocabulary.
    input_cue : nengo.Node
        Cue input.
    input_target : nengo.Node
        Target to learn.
    input_no_learn : nengo.Node
        Inhibits learning with an input of 1.
    output : nengo.Node
        Output of associated vector.
    """
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
            self.input_lr = nengo.Node(size_in=1)

            for i, e in enumerate(self.state.all_ensembles):
                sd = e.dimensions
                start = i * sd
                end = (i + 1) * sd
                conn = nengo.Connection(
                    e, self.output[start:end],
                    learning_rule_type=AML(sd, 10.),
                    function=lambda x, sd=sd: np.zeros(sd),  # noqa, pylint: disable=undefined-variable
                    solver=nengo.solvers.LstsqL2(solver=RandomizedSVD()))
                n = nengo.Node(size_in=sd + 1)
                nengo.Connection(
                    self.target.output[start:end], n[1:])
                nengo.Connection(self.input_lr, n[0])
                nengo.Connection(n, conn.learning_rule)

            self.compare = SimilarityThreshold(self.output_vocab)
            nengo.Connection(self.output, self.compare.input_a)
            nengo.Connection(self.input_target, self.compare.input_b)
            inhibit_net(self.compare.output, self.target, strength=1.)

            if init_transform is not None:
                nengo.Connection(
                    self.state.output, self.output, transform=init_transform)

            self.input_no_learn = nengo.Node(size_in=1)
            inhibit_net(self.input_no_learn, self.target)

        self.inputs = {
            'default': (self.input_cue, self.input_vocab),
            'target': (self.input_target, self.output_vocab),
            'no_learn': (self.input_no_learn, None)}
        self.outputs = {'default': (self.output, self.output_vocab)}


class Context(spa.Network):
    """Network to store and update context in TCM fashion.

    Parameters
    ----------
    beta : float
        TCM beta parameter, the amount of context drift with each item.
    vocab : nengo_spa.Vocabulary or int
        Vocabulary to use.
    kwargs : dict
        Passed on to `nengo_spa.Network`.

    Attributes
    ----------
    vocab : nengo_spa.Vocabulary
        Vocabulary to use.
    input : nengo.Node
        Input used to update context.
    input_update_context : nengo.Node
        Control signal when to update context.
    output : nengo.Node
        Output of current context.
    """

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
            self.update_ctx_invert = nengo.Ensemble(25, 1)
            nengo.Connection(self.bias, self.update_ctx_invert)
            nengo.Connection(
                self.input_update_context, self.update_ctx_invert.neurons,
                transform=-2. * np.ones((self.update_ctx_invert.n_neurons, 1)))

            nengo.Connection(
                self.update_ctx_invert, self.current.input_store,
                synapse=.005)
            nengo.Connection(
                self.input_update_context, self.old.input_store, synapse=0.005)

        self.output = self.current.output
        self.inputs = dict(default=(self.input, vocab),
                           update_context=(self.input_update_context, None))
        self.outputs = dict(default=(self.output, vocab))


class NeuralAccumulatorDecisionProcess(spa.Network):
    """Neural independent accumulator decision process for recall.

    Parameters
    ----------
    vocab : nengo_spa.Vocabulary or int
        Vocabulary to use for recallable pointers.
    noise : float
        Amount of noise to add to the input.
    kwargs : dict
        Passed on to `nengo_spa.Network`.

    Attributes
    ----------
    vocab : nengo_spa.Vocabulary
        Vocabulary to use for recallable pointers.
    input : nengo.Node
        Input of retrieved vector.
    output : nengo.Node
        Recalled vector.
    """
    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    # pylint: disable=too-many-statements
    def __init__(
            self, vocab=Default, noise=0., min_evidence=0., n_inputs=1,
            **kwargs):
        super(NeuralAccumulatorDecisionProcess, self).__init__(**kwargs)

        self.vocab = vocab

        n_items, d = self.vocab.vectors.shape

        with self:
            assert n_inputs > 0
            self.inputs = [nengo.Node(size_in=d) for _ in range(n_inputs)]

            # Input rectification
            with nengo.presets.ThresholdingEnsembles(0.):
                self.inp_thrs = [
                    nengo.networks.EnsembleArray(50, n_items)
                    for _ in range(n_inputs)]

            # Evidence integration
            with nengo.presets.ThresholdingEnsembles(0.):
                self.state = nengo.networks.EnsembleArray(50, n_items + 1)

            for inp, inp_thr in zip(self.inputs, self.inp_thrs):
                nengo.Connection(
                    inp, inp_thr.input,
                    transform=self.vocab.vectors, synapse=None)
                nengo.Connection(
                    inp_thr.output, self.state.input[:-1], transform=0.1)

            self.bias = nengo.Node(1.)
            nengo.Connection(
                self.bias, self.state.input[-1], transform=min_evidence)
            nengo.Connection(self.state.output, self.state.input, synapse=0.1)

            # Thresholding layer
            with nengo.presets.ThresholdingEnsembles(0.8):
                self.threshold = nengo.networks.EnsembleArray(50, n_items+1)
            nengo.Connection(self.state.output, self.threshold.input)
            tr = -2 * (1. - np.eye(n_items+1)) + 1. * np.eye(n_items + 1)
            nengo.Connection(
                self.threshold.add_output(
                    'heaviside', lambda x: 1 if x > 0.8 else 0.),
                self.state.input,
                transform=tr,
                synapse=0.1)

            # Buffer for recalled item
            self.buf = GatedMemory(self.vocab)
            nengo.Connection(
                self.threshold.heaviside[:-1], self.buf.diff.input,
                transform=self.vocab.vectors.T)
            self.buf_input_store = nengo.Ensemble(25, 1)
            nengo.Connection(self.buf_input_store, self.buf.input_store)
            nengo.Connection(nengo.Node(1.), self.buf_input_store)
            nengo.Connection(
                self.threshold.heaviside[:-1], self.buf_input_store.neurons,
                transform=-2. * np.ones(
                    (self.buf_input_store.n_neurons, n_items)))

            # Inhibition of recalled items
            self.inhibit = spa.State(self.vocab, feedback=1.)
            self.inhibit_gate = spa.State(self.vocab)
            nengo.Connection(
                self.buf.mem.output, self.inhibit_gate.input)
            nengo.Connection(
                self.inhibit_gate.output, self.inhibit.input, synapse=0.1,
                transform=0.1)
            self.cmp = spa.Compare(self.vocab, neurons_per_dimension=50)
            nengo.Connection(self.buf.mem.output, self.cmp.input_a)
            nengo.Connection(self.inhibit.output, self.cmp.input_b)
            with nengo.presets.ThresholdingEnsembles(0.):
                self.inhibit_update_done = nengo.Ensemble(50, 1)
            nengo.Connection(nengo.Node(-0.5), self.inhibit_update_done)
            nengo.Connection(self.cmp.output, self.inhibit_update_done)
            inhibit_net(
                self.inhibit_update_done, self.inhibit_gate,
                function=lambda x: x > 0)
            with nengo.presets.ThresholdingEnsembles(0.1):
                self.inhib_thr = nengo.networks.EnsembleArray(50, n_items)
            nengo.Connection(
                self.inhibit.output, self.inhib_thr.input,
                transform=self.vocab.vectors)
            nengo.Connection(
                self.inhib_thr.output, self.state.input[:-1], transform=-6.)

            # Start over if recall fails
            self.failed_recall_int = nengo.Ensemble(50, 1)
            nengo.Connection(
                self.failed_recall_int, self.failed_recall_int, synapse=0.1,
                transform=0.9)
            nengo.Connection(
                self.threshold.heaviside[-1], self.failed_recall_int,
                transform=0.1, synapse=0.1)
            with nengo.presets.ThresholdingEnsembles(0.):
                self.failed_recall = nengo.Ensemble(50, 1)
            nengo.Connection(self.failed_recall_int, self.failed_recall)
            nengo.Connection(nengo.Node(1.), self.failed_recall, transform=-0.3)
            self.failed_recall_heaviside = nengo.Node(size_in=1)
            nengo.Connection(
                self.failed_recall, self.failed_recall_heaviside,
                function=lambda x: x > 0.)
            for e in self.state.ensembles:
                nengo.Connection(
                    self.failed_recall_heaviside, e.neurons,
                    transform=-50. * np.ones((e.n_neurons, 1)), synapse=0.1)

            # Noise on input
            if noise > 0.:
                self.noise = nengo.Node(nengo.processes.WhiteNoise(
                    dist=nengo.dists.Gaussian(mean=0., std=noise)),
                                        size_out=n_items+1)
                nengo.Connection(self.noise, self.state.input)

            self.output = self.buf.output

        self.inputs = dict(default=(self.input[0], self.vocab))
        self.outputs = dict(default=(self.output, self.vocab))
