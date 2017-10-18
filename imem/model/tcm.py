from __future__ import absolute_import, print_function

import nengo
from nengo.config import Default
from nengo.utils.least_squares_solvers import RandomizedSVD
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam
import numpy as np

from imem.model.amlearn import AML
from imem.model.modules import GatedMemory, SimilarityThreshold
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
    def __init__(self, task_vocabs, beta, **kwargs):
        kwargs.setdefault('label', 'TCM')
        super(TCM, self).__init__(**kwargs)

        self.task_vocabs = task_vocabs

        with self:
            self.bias = nengo.Node(1.)

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
            nengo.Connection(self.current_ctx.output, self.net_m_tf.input_cue)

            # Control of learning
            self.input_no_learn = nengo.Node(size_in=1)
            nengo.Connection(self.input_no_learn, self.net_m_ft.input_no_learn)
            nengo.Connection(self.input_no_learn, self.net_m_tf.input_no_learn)

            # Initialization of context
            initial_ctx = self.task_vocabs.contexts.create_pointer().v
            ctx_init = nengo.Node(
                lambda t: initial_ctx if t < 0.3 else np.zeros(
                    self.task_vocabs.contexts.dimensions))
            nengo.Connection(ctx_init, self.current_ctx.current.input)
            nengo.Connection(
                nengo.Node(lambda t: 4 if t < 0.3 else 0),
                self.current_ctx.old.input_store)

            self.input_update_context = self.current_ctx.input_update_context
            self.output_recalled_item = self.net_m_tf.output

        self.inputs = dict(
            default=(self.input, self.task_vocabs.items),
            input_pos=(self.input_pos, self.task_vocabs.positions),
            input_update_context=(self.input_update_context, None),
            input_no_learn=(self.input_no_learn, None))
        self.outputs = dict(
            output_recalled_item=(
                self.output_recalled_item, self.task_vocabs.items))


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
            self.scale = nengo.Node(1.)

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
                nengo.Connection(self.scale, n[0])
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
        self.inputs = dict(
            default=(self.input, vocab),
            input_update_context=(self.input_update_context, None))
        self.outputs = dict(default=(self.output, vocab))
