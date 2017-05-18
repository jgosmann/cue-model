from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem import protocols
from imem import tcm


class NeuralTCM(pytry.NengoTrial):
    def params(self):
        self.param("List length to remember", n_items=12)
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=1.)
        self.param("protocol", protocol='immed')

    def model(self, p):
        with spa.Network(seed=p.seed) as model:
            model.config[spa.State].represent_identity = False

            proto = {
                'contdist': protocols.FreeRecall(
                    n_items=p.n_items, pi=1.2, ipi=16., ri=16.),
                'delayed': protocols.FreeRecall(
                    n_items=p.n_items, pi=1.2, ipi=0., ri=16.),
                'immed': protocols.FreeRecall(
                    n_items=p.n_items, pi=1., ipi=0., ri=0.),
            }[p.protocol]

            self.control = tcm.Control(
                proto, p.item_d, p.distractor_rate,
                np.random.RandomState(p.seed + 1))

            model.tcm = tcm.TCM(p.beta, self.control, p.item_d, p.context_d)

            self.p_recalls = nengo.Probe(
                model.tcm.recall.buf.mem.output, synapse=0.01)

        self._model = model
        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.control.protocol.duration + 60.)

        recall_vocab = self.control.vocab.create_subset(
            ['V' + str(i) for i in range(self.control.protocol.n_items)])
        similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
        responses = []
        for x in np.argmax(above_threshold, axis=1):
            if x not in responses:
                responses.append(float(x))
        responses = responses + (p.n_items - len(responses)) * [np.nan]

        return {
            'responses': responses,
            'vocab_vectors': self.control.vocab.vectors,
            'vocab_keys': self.control.vocab.keys(),
        }
