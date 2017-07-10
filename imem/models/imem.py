from __future__ import absolute_import

from functools import partial

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem import protocols
from imem.imem import IMem, Vocabularies


class IMemTrial(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    PROTOCOLS = {
        'contdist': partial(protocols.Recall, pi=1.2, ipi=16., ri=16.),
        'delayed': partial(protocols.Recall, pi=1.2, ipi=0., ri=16.),
        'immed': partial(protocols.Recall, pi=1., ipi=0., ri=0.),
        'serial': partial(protocols.Recall, pi=1., ipi=0., ri=0., serial=True),
    }

    @classmethod
    def get_proto(cls, p):
        return cls.PROTOCOLS[p.protocol](
            n_items=p.n_items, distractor_rate=p.distractor_rate)

    def params(self):
        self.param("List length to remember", n_items=12)
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("distractor rate", distractor_rate=1.)
        self.param("noise in recall", noise=0.)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        proto = self.get_proto(p)
        self.vocabs = Vocabularies(proto, p.item_d, p.context_d, p.n_items + 3)

        with spa.Network(seed=p.seed) as model:
            model.config[spa.State].represent_identity = False

            model.imem = IMem(proto, self.vocabs, p.beta, p.gamma, p.noise)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)

            # self.p_ose = nengo.Probe(model.imem.ose.output, synapse=0.01)
            self.p_ose_store = nengo.Probe(
                model.imem.ose.input_store, synapse=0.01)
            self.p_input_item = nengo.Probe(
                model.imem.ose.input_item, synapse=0.01)
            self.p_input_pos = nengo.Probe(
                model.imem.ose.input_pos, synapse=0.01)
            self.p_g1 = nengo.Probe(
                model.imem.ose.combine.diff.output, synapse=0.01)
            self.p_g2 = nengo.Probe(
                model.imem.ose.mem.diff.output, synapse=0.01)
            self.p_combine = nengo.Probe(
                model.imem.ose.combine.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.imem.pos.output, synapse=0.01)
            self.p_bind = nengo.Probe(model.imem.ose.bind.output, synapse=0.01)
            self.p_bind_a = nengo.Probe(model.imem.ose.bind.input_a, synapse=0.01)
            self.p_bind_b = nengo.Probe(model.imem.ose.bind.input_b, synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        proto = self.get_proto(p)

        sim.run(proto.duration + p.recall_duration)

        recall_vocab = self.vocabs.items.create_subset(proto.get_all_items())
        similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
        responses = []
        for x in np.argmax(above_threshold, axis=1):
            if x not in responses:
                responses.append(float(x))
        responses = responses + (p.n_items - len(responses)) * [np.nan]

        result = {
            'responses': responses,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
            # 'ose': sim.data[self.p_ose],
            'ose_store': sim.data[self.p_ose_store],
            'input_item': sim.data[self.p_input_item],
            'input_pos': sim.data[self.p_input_pos],
            'g1': sim.data[self.p_g1],
            'g2': sim.data[self.p_g2],
            'combine': sim.data[self.p_combine],
            'pos': sim.data[self.p_pos],
            'bind': sim.data[self.p_bind],
            'bind_a': sim.data[self.p_bind_a],
            'bind_b': sim.data[self.p_bind_b],
        }
        # if p.debug:
            # np.savez('debug.npz', **result)
        return result
