from __future__ import absolute_import, division, print_function

import nengo
from nengo.config import Config
from nengo.params import FrozenObject
from nengo.utils.compat import is_integer
import nengo_spa as spa
from nengo_spa.vocab import (
    VocabularyMap, VocabularyMapParam, VocabularyOrDimParam)
import numpy as np

from imem.model.modules import SimilarityThreshold
from imem.model.networks import OneHotCounter
from imem.model.ose import OSE
from imem.model.recall import NeuralAccumulatorDecisionProcess
from imem.model.tcm import TCM
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
    output_learn : nengo.Ensemble
        Outputs 1 when learning should be enabled.
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
                protocol.proto.serial, label='output_serial_recall')
            self.output_free_recall = nengo.Node(
                not protocol.proto.serial, label='output_serial_recall')

            self._current_stim = None
            self.output_no_learn = nengo.Node(
                lambda t: (self.protocol.is_recall_phase(t) or
                           self._current_stim is None or
                           self._current_stim.startswith('D')),
                label='output_no_learn')
            self.output_no_pos_count = nengo.Node(
                lambda t: (
                    (self.protocol.is_recall_phase(t) and
                     not self.protocol.proto.serial) or
                    self._current_stim is None or
                    (self._current_stim.startswith('D') and
                     not protocol.proto.serial)),
                label='output_no_learn')

            self.bias = nengo.Node(1.)
            self.output_learn = nengo.Ensemble(
                25, 1, encoders=nengo.dists.Choice([[1.]]))
            nengo.Connection(self.bias, self.output_learn)
            nengo.Connection(
                self.output_no_learn, self.output_learn, transform=-1.)

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

    def __init__(self, stim_provider, items, contexts, n_pos, rng=None):
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

        self.items.populate(';'.join(stim_provider.get_all_items()))
        if stim_provider.n_distractors_per_epoch > 0:
            self.items.populate(';'.join(stim_provider.get_all_distractors()))

        for i in range(self.items.dimensions):
            self.contexts.populate('CTX' + str(i))

        for i in range(n_pos + 3):
            self.positions.populate('P' + str(i))


class IMem(spa.Network):
    def __init__(
            self, protocol, task_vocabs, beta, gamma=0.9775, ose_thr=0.2,
            ordinal_prob=0.2, recall_noise=0., **kwargs):
        kwargs.setdefault('label', 'IMem')
        super(IMem, self).__init__(**kwargs)

        self.task_vocabs = task_vocabs

        with self:
            self.config[spa.State].represent_identity = False

            self.bias = nengo.Node(1.)
            self.ctrl = Control(protocol, self.task_vocabs.items)

            # TCM
            self.tcm = TCM(self.task_vocabs, beta)
            nengo.Connection(self.ctrl.output_stimulus, self.tcm.input)

            # position counter
            self.pos = OneHotCounter(len(self.task_vocabs.positions))
            nengo.Connection(
                self.ctrl.output_no_pos_count, self.pos.rising_edge_gate,
                transform=-1.)
            nengo.Connection(nengo.Node(lambda t: t < 0.3), self.pos.input[0])

            # Short term memory
            self.ose = OSE(self.task_vocabs.items, gamma)
            nengo.Connection(self.ctrl.output_stimulus, self.ose.input_item)

            # primacy effect
            # FIXME time dependence for different protocols
            nengo.Connection(
                nengo.Node(
                    lambda t: -np.exp(-t / 1.) if t < 12. else 0.),
                self.tcm.net_m_tf.compare.threshold)

            # Use irrelevant position vector to bind distractors
            self.in_pos_gate = spa.State(self.task_vocabs.positions)
            nengo.Connection(self.pos.output, self.in_pos_gate.input,
                             transform=self.task_vocabs.positions.vectors.T)
            nengo.Connection(self.in_pos_gate.output, self.ose.input_pos)
            nengo.Connection(self.in_pos_gate.output, self.tcm.input_pos)

            self.irrelevant_pos_gate = spa.State(self.task_vocabs.positions)
            self.irrelevant_pos = nengo.Node(
                self.task_vocabs.positions.create_pointer().v)
            nengo.Connection(self.irrelevant_pos,
                             self.irrelevant_pos_gate.input)
            nengo.Connection(
                self.irrelevant_pos_gate.output, self.ose.input_pos)

            with nengo.presets.ThresholdingEnsembles(0.):
                self.in_pos_gate_inhibit = nengo.Ensemble(25, 1)
            inhibit_net(self.in_pos_gate_inhibit, self.in_pos_gate)
            nengo.Connection(
                self.ctrl.output_no_learn, self.in_pos_gate_inhibit)
            nengo.Connection(
                self.ctrl.output_recall_phase, self.in_pos_gate_inhibit,
                transform=-1)
            inhibit_net(self.ctrl.output_learn, self.irrelevant_pos_gate)
            inhibit_net(
                self.ctrl.output_recall_phase, self.irrelevant_pos_gate)
            nengo.Connection(
                self.irrelevant_pos_gate.output, self.tcm.input_pos)

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

            # Certain fraction of recalls use ordinal strategy
            if np.random.rand() >= ordinal_prob:
                nengo.Connection(
                    self.ctrl.output_free_recall, self.start_of_recall,
                    transform=-5.)

            # Determining update progress
            self.last_item = spa.State(self.task_vocabs.items, feedback=1.)
            self.sim_th = SimilarityThreshold(self.task_vocabs.items)
            nengo.Connection(
                self.ctrl.output_stimulus, self.last_item.input, transform=1.,
                synapse=0.1)
            nengo.Connection(self.ctrl.output_stimulus, self.sim_th.input_a)
            nengo.Connection(self.last_item.output, self.sim_th.input_b)
            nengo.Connection(self.bias, self.tcm.input_update_context)
            nengo.Connection(
                self.sim_th.output, self.tcm.input_update_context,
                transform=-1.)

            nengo.Connection(
                self.sim_th.output, self.pos.input_inc, transform=-1)
            nengo.Connection(self.sim_th.output, self.ose.input_store)

            nengo.Connection(self.bias, self.tcm.input_no_learn)
            nengo.Connection(
                self.sim_th.output, self.tcm.input_no_learn, transform=-1)
            nengo.Connection(
                self.ctrl.output_no_learn, self.tcm.input_no_learn,
                transform=2.)

            # Recall networks
            self.recall = NeuralAccumulatorDecisionProcess(
                self.task_vocabs.items.create_subset(protocol.get_all_items()),
                noise=recall_noise, min_evidence=.025, n_inputs=2)
            self.recalled_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(self.recalled_gate.output, self.tcm.input)

            self.pos_recall = NeuralAccumulatorDecisionProcess(
                self.task_vocabs.positions, noise=recall_noise)
            self.pos_recalled_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(
                self.pos_recalled_gate.output, self.tcm.input_pos)

            self.tcm_recall_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(
                self.tcm.output_recalled_item, self.tcm_recall_gate.input)
            inhibit_net(self.ctrl.output_pres_phase, self.tcm_recall_gate)

            for recall_net, recalled_gate in (
                    (self.recall, self.recalled_gate),
                    (self.pos_recall, self.pos_recalled_gate)):
                nengo.Connection(
                    self.tcm_recall_gate.output, recall_net.input_list[0])
                inhibit_net(
                    self.ctrl.output_serial_recall, recalled_gate, strength=6)
                nengo.Connection(recall_net.output, recalled_gate.input)
                nengo.Connection(recall_net.output, self.sim_th.input_a)
                nengo.Connection(
                    recall_net.output, self.last_item.input, transform=0.1,
                    synapse=0.1)

                inhibit_net(self.ctrl.output_pres_phase, recall_net.state)
                inhibit_net(
                    self.ctrl.output_pres_phase, recall_net.buf.mem,
                    strength=6)
                inhibit_net(self.ctrl.output_pres_phase, recall_net.inhibit)

            # on failed recall increment pos and update context
            nengo.Connection(
                self.recall.failed_recall_heaviside, self.pos.input_inc,
                transform=50., synapse=0.1)
            nengo.Connection(
                self.recall.failed_recall_heaviside,
                self.tcm.input_update_context, transform=20.)

            # Set position from recalled positions
            self.pos_gate = nengo.networks.EnsembleArray(
                30, len(self.task_vocabs.positions))
            nengo.Connection(
                self.pos_recall.buf.output, self.pos_gate.input,
                transform=1.3 * self.task_vocabs.positions.vectors)
            nengo.Connection(
                self.bias, self.pos_gate.input,
                transform=-np.ones((self.pos_gate.input.size_in, 1)))
            nengo.Connection(
                self.pos_gate.output, self.pos.input, transform=10.,
                synapse=0.1)
            self.invert = nengo.Ensemble(30, 1)
            nengo.Connection(self.bias, self.invert)
            nengo.Connection(
                self.pos_recall.buf.mem.state_ensembles.add_output(
                    'square', lambda x: x * x),
                self.invert,
                transform=-np.ones((1, self.task_vocabs.positions.dimensions)))
            inhibit_net(self.invert, self.pos_gate)
            inhibit_net(self.ctrl.output_serial_recall, self.pos_gate)
            inhibit_net(self.ctrl.output_serial_recall, self.pos_recall.buf)

            # Short term recall
            self.ose_recall_gate = spa.State(self.task_vocabs.items)
            nengo.Connection(self.ose.output, self.ose_recall_gate.input)
            nengo.Connection(
                self.ose_recall_gate.output, self.recall.input_list[1])
            self.ose_recall_threshold = nengo.Node(ose_thr)
            nengo.Connection(
                self.ose_recall_threshold, self.recall.inp_thrs[1].input,
                transform=-np.ones(
                    (self.recall.inp_thrs[1].input.size_in, 1)))
            inhibit_net(self.ctrl.output_pres_phase, self.ose_recall_gate)
            inhibit_net(self.start_of_recall, self.ose_recall_gate)
            inhibit_net(self.start_of_recall, self.tcm.current_ctx.old.mem,
                        synapse=0.1, strength=5)

            self.output = self.recall.output
            self.output_pos = self.pos.output

        self.outputs = dict(
            default=(self.output, self.task_vocabs.items),
            output_pos=(self.output_pos, self.task_vocabs.positions))
