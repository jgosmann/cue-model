import nengo
from nengo.config import Default
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

from imem.utils.nengo import inhibit_net


class GatedMemory(spa.Network):
    """Memory module with gated input.

    Provide an input of 1 to ``store`` to preserve the current value in memory
    and a value of 0 to overwrite it with the current input.
    """

    vocab = VocabularyOrDimParam('vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, in_tr=1, in_syn=0.1, **kwargs):
        super(GatedMemory, self).__init__(**kwargs)

        self.vocab = vocab

        with self:
            self.diff = spa.State(self.vocab)
            self.mem = spa.State(self.vocab, feedback=1)
            self.store = nengo.Node(size_in=1)

            nengo.Connection(
                self.diff.output, self.mem.input, transform=in_tr / in_syn,
                synapse=in_syn)
            nengo.Connection(
                self.mem.output, self.diff.input, transform=-1)
            inhibit_net(self.store, self.diff.state_ensembles, strength=3.)

            self.input = self.diff.input
            self.output = self.mem.output

        self.inputs = dict(default=(self.diff.input, vocab))
        self.outputs = dict(default=(self.mem.output, vocab))


class SimilarityThreshold(spa.Network):
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
