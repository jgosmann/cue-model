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
