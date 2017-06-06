from __future__ import division

import nengo
from nengo.config import Default
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

from imem.utils.nengo import inhibit_net


class GatedMemory(spa.Network):
    """Memory module with gated input.

    Provide an input of 1 to ``input_store`` to preserve the current value in
    memory and a value of 0 to overwrite it with the current input.

    Parameters
    ----------
    vocab : nengo_spa.Vocabulary or int
        Vocabulary.
    feedback_syn : float
        Synaptic time constant of the feedback connection.

    Attributes
    ----------
    vocab : nengo_spa.Vocabulary
        Vocabulary.
    input : nengo.Node
        Input node.
    output : nengo.Node
        Output node.
    input_store : nengo.Node
        Input 1 to preserve current value and 0 to update current value.
    """

    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, feedback_syn=.1, **kwargs):
        super(GatedMemory, self).__init__(**kwargs)

        self.vocab = vocab

        with self:
            self.diff = spa.State(self.vocab)
            self.mem = spa.State(self.vocab, feedback=1)
            self.input_store = nengo.Node(size_in=1)

            nengo.Connection(
                self.diff.output, self.mem.input, transform=1. / feedback_syn,
                synapse=feedback_syn)
            nengo.Connection(
                self.mem.output, self.diff.input, transform=-1)
            inhibit_net(
                self.input_store, self.diff.state_ensembles, strength=3.)

            self.input = self.diff.input
            self.output = self.mem.output

        self.inputs = dict(
            default=(self.diff.input, vocab), store=(self.input_store, None))
        self.outputs = dict(default=(self.mem.output, vocab))


class SimilarityThreshold(spa.Network):
    """Outputs 1 when the dot product of the inputs exceeds threshold.

    Parameters
    ----------
    vocab : nengo_spa.Vocabulary or int
        Input vocabulary.
    threshold : float
        Threshold that needs to be exceeded to produce an output.
    kwargs : dict
        Passed on to `nengo_spa.Network`.

    Attributes
    ----------
    vocab : nengo_spa.Vocabulary
        Input vocabulary.
    input_a : nengo.Node
        First input.
    input_b : nengo.Node
        Second input.
    output : nengo.Node
        Output.
    """

    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, threshold=1., **kwargs):
        super(SimilarityThreshold, self).__init__(**kwargs)

        self.vocab = vocab

        with self:
            self.bias = nengo.Node(1)
            self.dot = spa.Compare(self.vocab)
            with nengo.presets.ThresholdingEnsembles(0.0):
                self.threshold = nengo.Ensemble(150, 1)
            nengo.Connection(self.bias, self.threshold, transform=-threshold)

            self.output = nengo.Node(size_in=1)
            self.input_a = self.dot.input_a
            self.input_b = self.dot.input_b

            nengo.Connection(self.dot.output, self.threshold)
            nengo.Connection(
                self.threshold, self.output, function=lambda x: x > 0.,
                synapse=None)

        self.inputs = dict(
            input_a=(self.input_a, self.vocab),
            input_b=(self.input_b, self.vocab))
        self.outputs = dict(default=(self.output, None))
