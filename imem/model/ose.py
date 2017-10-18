import nengo
from nengo.config import Default
import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

from imem.model.modules import GatedMemory


class OSE(spa.Network):
    vocab = VocabularyOrDimParam(
        'item_vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, gamma=0.9775, **kwargs):
        kwargs.setdefault('label', 'OSE')
        super(OSE, self).__init__(**kwargs)

        self.vocab = vocab

        with self:
            self.input_item = nengo.Node(size_in=self.vocab.dimensions)
            self.input_pos = nengo.Node(size_in=self.vocab.dimensions)
            self.input_store = nengo.Node(size_in=1)

            self.bind = spa.Bind(self.vocab)
            nengo.Connection(self.input_item, self.bind.input_a)
            nengo.Connection(self.input_pos, self.bind.input_b)

            with nengo.Config(spa.State) as cfg:
                cfg[spa.State].neurons_per_dimension = 150
                cfg[spa.State].subdimensions = 4

                self.combine = GatedMemory(vocab)
                nengo.Connection(self.bind.output, self.combine.input)

                self.mem = GatedMemory(vocab)
                nengo.Connection(self.combine.output, self.mem.input)
                nengo.Connection(
                    self.mem.output, self.combine.input, transform=gamma)

            nengo.Connection(self.input_store, self.combine.input_store)
            nengo.Connection(
                self.input_store, self.mem.input_store, transform=-1)
            self.bias = nengo.Node(1.)
            nengo.Connection(self.bias, self.mem.input_store)

            self.recall = spa.Bind(self.vocab, invert_a=True)
            nengo.Connection(self.input_pos, self.recall.input_a)
            nengo.Connection(self.mem.output, self.recall.input_b)
            self.output = self.recall.output

        self.inputs = dict(
            input_item=(self.input_item, self.vocab),
            input_pos=(self.input_pos, self.vocab),
            input_store=(self.input_store, None))
        self.outputs = dict(default=(self.output, self.vocab))
