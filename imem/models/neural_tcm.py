from __future__ import absolute_import

from functools import partial

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem import protocols
from imem import tcm


class NeuralTCM(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    PROTOCOLS = {
        'contdist': partial(protocols.FreeRecall, pi=1.2, ipi=16., ri=16.),
        'delayed': partial(protocols.FreeRecall, pi=1.2, ipi=0., ri=16.),
        'immed': partial(protocols.FreeRecall, pi=1., ipi=0., ri=0.),
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
        with spa.Network(seed=p.seed) as model:
            model.config[spa.State].represent_identity = False

            proto = self.get_proto(p)
            model.tcm = tcm.TCM(p.beta, proto, p.noise, p.item_d, p.context_d)
            self.p_recalls = nengo.Probe(model.tcm.output, synapse=0.01)

        self._model = model
        return model

    def evaluate(self, p, sim, plt):
        proto = self.get_proto(p)

        sim.run(proto.duration + p.recall_duration)

        recall_vocab = self._model.tcm.item_vocab.create_subset(
            proto.get_all_items())
        similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
        responses = []
        for x in np.argmax(above_threshold, axis=1):
            if x not in responses:
                responses.append(float(x))
        responses = responses + (p.n_items - len(responses)) * [np.nan]

        return {
            'responses': responses,
            'vocab_vectors': self._model.tcm.item_vocab.vectors,
            'vocab_keys': list(self._model.tcm.item_vocab.keys()),
        }
