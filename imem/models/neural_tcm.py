from __future__ import absolute_import

from functools import partial

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem import protocols
from imem import tcm
from imem.imem import Control, Vocabularies


class NeuralTCM(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    PROTOCOLS = {
        'contdist': partial(protocols.Recall, pi=1.2, ipi=16., ri=16.),
        'delayed': partial(protocols.Recall, pi=1.2, ipi=0., ri=16.),
        'immed': partial(protocols.Recall, pi=1., ipi=0., ri=0.),
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
        self.param("distractor rate", distractor_rate=1.)
        self.param("noise in recall", noise=0.)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        proto = self.get_proto(p)
        self.vocabs = Vocabularies(proto, p.item_d, p.context_d, 0)

        with spa.Network(seed=p.seed) as model:
            model.config[spa.State].represent_identity = False

            model.control = Control(proto, self.vocabs.items)
            model.tcm = tcm.TCM(
                self.vocabs, proto, model.control, p.beta, p.noise)
            self.p_recalls = nengo.Probe(model.tcm.output, synapse=0.01)

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
        }
        return result
