import nengo
from nengo.config import Default
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam
import numpy as np

from imem.model.modules import GatedMemory
from imem.utils.nengo import inhibit_net


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
            self.input_list = [nengo.Node(size_in=d) for _ in range(n_inputs)]

            # Input rectification
            with nengo.presets.ThresholdingEnsembles(0.):
                self.inp_thrs = [
                    nengo.networks.EnsembleArray(50, n_items)
                    for _ in range(n_inputs)]

            # Evidence integration
            with nengo.presets.ThresholdingEnsembles(0.):
                self.state = nengo.networks.EnsembleArray(50, n_items + 1)

            for inp, inp_thr in zip(self.input_list, self.inp_thrs):
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
            self.buf = GatedMemory(self.vocab, diff_scale=10.)
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
            # nengo.Connection(
                # self.inhibit.output, self.inhib_thr.input,
                # transform=self.vocab.vectors)
            self.out_inhibit_gate = GatedMemory(self.vocab)
            nengo.Connection(self.inhibit.output, self.out_inhibit_gate.input)
            nengo.Connection(
                self.out_inhibit_gate.output, self.inhib_thr.input,
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
            nengo.Connection(
                nengo.Node(1.), self.failed_recall, transform=-0.3)
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

        self.inputs = dict(default=(self.input_list[0], self.vocab))
        self.outputs = dict(default=(self.output, self.vocab))
