from __future__ import absolute_import, division, print_function

import nengo
from nengo.config import Config, Default
from nengo.params import FrozenObject
from nengo.utils.compat import is_integer
import nengo_spa as spa
from nengo_spa.vocab import (
    VocabularyMap, VocabularyMapParam, VocabularyOrDimParam)
import numpy as np

from imem.modules import GatedMemory
from imem.networks import OneHotCounter
from imem.tcm import TCM
from imem.utils.nengo import inhibit_net


class Control(nengo.Network):
    """Non-neural control signals for the integrated memory model.

    Parameters
    ----------
    protocol : Recall
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
            self.output_serial_recall = nengo.Node(
                protocol.serial, label='output_serial_recall')
            self.output_free_recall = nengo.Node(
                not protocol.serial, label='output_serial_recall')

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


class Vocabularies(FrozenObject):
    vocabs = VocabularyMapParam(
        'vocabs', default=None, optional=False, readonly=True)

    items = VocabularyOrDimParam(
        'items', optional=False, readonly=True)
    contexts = VocabularyOrDimParam(
        'contexts', optional=False, readonly=True)
    positions = VocabularyOrDimParam(
        'positions', optional=False, readonly=True)

    def __init__(self, protocol, items, contexts, n_pos, rng=None):
        super(Vocabularies, self).__init__()

        vocabs = Config.default(spa.Network, 'vocabs')
        if vocabs is None:
            vocabs = VocabularyMap(rng=rng)
        self.vocabs = vocabs

        self.items = items
        if is_integer(contexts):
            contexts = spa.Vocabulary(contexts)
        self.contexts = contexts
        self.positions = spa.Vocabulary(self.items.dimensions)

        self.items.populate(';'.join(protocol.get_all_items()))
        if protocol.n_distractors_per_epoch > 0:
            self.items.populate(';'.join(protocol.get_all_distractors()))

        for i in range(self.items.dimensions):
            self.contexts.populate('CTX' + str(i))

        for i in range(n_pos + 3):
            self.positions.populate('P' + str(i))


class OSE(spa.Network):
    vocab = VocabularyOrDimParam(
        'item_vocab', optional=False, readonly=True)

    def __init__(self, vocab=Default, gamma=0.9775, recall=False, **kwargs):
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

            if recall:
                self.recall = spa.Bind(self.vocab, invert_a=True)
                nengo.Connection(self.input_pos, self.recall.input_a)
                nengo.Connection(self.mem.output, self.recall.input_b)
                # self.cleanup = spa.ThresholdingAssocMem(
                    # threshold=0.3, input_vocab=self.vocab)
                # nengo.Connection(self.recall.output, self.cleanup.input)
                # self.output = self.cleanup.output
                self.output = self.recall.output
                self.outputs = dict(default=(self.output, self.vocab))

        self.inputs = dict(
            input_item=(self.input_item, self.vocab),
            input_pos=(self.input_pos, self.vocab),
            input_store=(self.input_store, None))


class IMem(spa.Network):
    def __init__(
            self, protocol, task_vocabs, beta, gamma=0.9775, recall_noise=0.,
            **kwargs):
        kwargs.setdefault('label', 'IMem')
        super(IMem, self).__init__(**kwargs)

        self.task_vocabs = task_vocabs

        with self:
            self.ctrl = Control(protocol, self.task_vocabs.items)

            self.tcm = TCM(
                self.task_vocabs, protocol, self.ctrl, beta,
                recall_noise=recall_noise)
            nengo.Connection(self.ctrl.output_stimulus, self.tcm.input)

            self.pos = OneHotCounter(len(self.task_vocabs.positions))
            nengo.Connection(self.tcm.output_stim_update_done,
                             self.pos.input_inc, transform=-1)
            nengo.Connection(nengo.Node(lambda t: t < 0.3), self.pos.input[0])
            nengo.Connection(self.pos.output, self.tcm.input_pos,
                             transform=self.task_vocabs.positions.vectors.T)

            # Reset of position
            with nengo.presets.ThresholdingEnsembles(0.):
                self.start_of_recall = nengo.Ensemble(50, 1)
            nengo.Connection(
                self.ctrl.output_recall_phase, self.start_of_recall,
                synapse=0.05, transform=-1)
            nengo.Connection(
                self.ctrl.output_recall_phase, self.start_of_recall,
                synapse=0.005)
            tr = -9 * np.ones((self.pos.input.size_in, 1))
            tr[0, 0] = 3.
            nengo.Connection(
                self.start_of_recall, self.pos.input, transform=tr,
                synapse=0.1)
            nengo.Connection(
                self.start_of_recall, self.tcm.input_pos,
                transform=np.atleast_2d(
                    self.task_vocabs.positions.vectors[0]).T,
                synapse=0.1)

            self.ose = OSE(
                self.task_vocabs.items, gamma, recall=protocol.serial)
            nengo.Connection(self.ctrl.output_stimulus, self.ose.input_item)
            nengo.Connection(self.pos.output, self.ose.input_pos,
                             transform=self.task_vocabs.positions.vectors.T)
            nengo.Connection(self.tcm.output_stim_update_done,
                             self.ose.input_store)
            if protocol.serial:
                self.ose_recall_gate = spa.State(self.task_vocabs.items)
                nengo.Connection(self.ose.output, self.ose_recall_gate.input)
                nengo.Connection(
                    self.ose_recall_gate.output, self.tcm.recall.input)
                inhibit_net(self.ctrl.output_pres_phase, self.ose_recall_gate)
                # FIXME additions
                inhibit_net(self.start_of_recall, self.ose_recall_gate)
                inhibit_net(self.start_of_recall, self.tcm.current_ctx.old.mem,
                            synapse=0.1, strength=5)

            self.output = self.tcm.output

        self.outputs = dict(default=(self.output, self.task_vocabs.items))
