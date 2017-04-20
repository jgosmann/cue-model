import nengo
import nengo_spa as spa

from imem.utils.nengo import inhibit_net


class GatedMemory(spa.Network):
    """Memory module with gated input.

    Provide an input of 1 to ``store`` to preserve the current value in memory
    and a value of 0 to overwrite it with the current input.
    """
    def __init__(
            self, d, vocab=None, in_tr=1, in_syn=0.005, label=None, seed=None,
            add_to_container=None):
        super(GatedMemory, self).__init__(label, seed, add_to_container)

        if vocab is None:
            vocab = d

        with self:
            self.diff = spa.State(d)
            self.mem = spa.State(d, feedback=1)#, subdimensions=1)
            self.store = nengo.Node(size_in=1)

            nengo.Connection(
                self.diff.output, self.mem.input, transform=in_tr,
                synapse=in_syn)
            nengo.Connection(
                self.mem.output, self.diff.input, transform=-1)
            inhibit_net(self.store, self.diff.state_ensembles, strength=3.)

        self.inputs = dict(default=(self.diff.input, vocab))
        self.outputs = dict(default=(self.mem.output, vocab))
