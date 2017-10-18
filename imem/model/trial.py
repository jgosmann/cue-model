from __future__ import absolute_import

from functools import partial

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem import protocols
from imem.model import IMem, Vocabularies


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
        self.param("distractor rate", distractor_rate=1.)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.2)
        self.param("TCM prob. to recall from beginning", ordinal_prob=0.2)
        self.param("noise in recall", noise=0.)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        proto = self.get_proto(p)
        self.vocabs = Vocabularies(proto, p.item_d, p.context_d, p.n_items + 3)

        with spa.Network(seed=p.seed) as model:
            model.config[spa.State].represent_identity = False
            # model.config[spa.State].neurons_per_dimension = 10
            # model.config[spa.Bind].neurons_per_dimension = 100
            # model.config[spa.Compare].neurons_per_dimension = 100

            model.imem = IMem(
                proto, self.vocabs, p.beta, p.gamma, p.noise,
                p.ose_thr, p.ordinal_prob)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)

            self.p_recall_state = nengo.Probe(model.imem.tcm.recall.state.output, synapse=0.01)
            self.p_recall_threshold = nengo.Probe(model.imem.tcm.recall.threshold.heaviside, synapse=0.01)
            self.p_recall_buf = nengo.Probe(model.imem.tcm.recall.buf.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.imem.pos.output, synapse=0.01)

            self.p_pos_recall_state = nengo.Probe(model.imem.tcm.pos_recall.state.output, synapse=0.01)
            self.p_pos_recall_buf = nengo.Probe(model.imem.tcm.pos_recall.buf.output, synapse=0.01)

            self.p_aml_comp = nengo.Probe(model.imem.tcm.net_m_tf.compare.output, synapse=0.01)
            self.p_ctx = nengo.Probe(model.imem.tcm.current_ctx.output, synapse=0.01)
            self.p_ctx_update = nengo.Probe(model.imem.tcm.current_ctx.input_update_context, synapse=0.01)
            self.p_inhib_recall = nengo.Probe(model.imem.tcm.recall.inhibit.output, synapse=0.01)
            self.p_recall_ctx = nengo.Probe(model.imem.tcm.net_m_ft.output, synapse=0.01)
            self.p_recall_ctx_cue = nengo.Probe(model.imem.tcm.net_m_ft.input_cue, synapse=0.01)

            self.p_input_pos = nengo.Probe(model.imem.tcm.input_pos, synapse=0.01)
            self.p_current_ctx = nengo.Probe(model.imem.tcm.current_ctx.output, synapse=0.01)
            self.p_input_update_ctx = nengo.Probe(model.imem.tcm.current_ctx.input_update_context, synapse=0.01)
            self.p_sim_th = nengo.Probe(model.imem.tcm.sim_th.output, synapse=0.01)
            self.p_last_item = nengo.Probe(model.imem.tcm.last_item.output, synapse=0.01)

            self.p_ose_output = nengo.Probe(model.imem.ose.output, synapse=0.01)
            self.p_tcm_output = nengo.Probe(model.imem.tcm.net_m_tf.output, synapse=0.01)

            self.p_failed_recall_int = nengo.Probe(model.imem.tcm.recall.failed_recall_int, synapse=0.01)
            self.p_failed_recall = nengo.Probe(model.imem.tcm.recall.failed_recall, synapse=0.01)
            self.p_failed_recall_heaviside = nengo.Probe(model.imem.tcm.recall.failed_recall_heaviside, synapse=0.01)

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
            'recall_state': sim.data[self.p_recall_state],
            'recall_threshold': sim.data[self.p_recall_threshold],
            'recall_buf': sim.data[self.p_recall_buf],
            'pos': sim.data[self.p_pos],
            'aml_comp': sim.data[self.p_aml_comp],
            'ctx': sim.data[self.p_ctx],
            'ctx_update': sim.data[self.p_ctx_update],
            'inhib_recall': sim.data[self.p_inhib_recall],
            'recall_ctx': sim.data[self.p_recall_ctx],
            'recall_ctx_cue': sim.data[self.p_recall_ctx_cue],
            'input_pos': sim.data[self.p_input_pos],
            'current_ctx': sim.data[self.p_current_ctx],
            'input_update_ctx': sim.data[self.p_input_update_ctx],
            'sim_th': sim.data[self.p_sim_th],
            'last_item': sim.data[self.p_last_item],
            'ose_output': sim.data[self.p_ose_output],
            'tcm_output': sim.data[self.p_tcm_output],
            'failed_recall_int': sim.data[self.p_failed_recall_int],
            'failed_recall': sim.data[self.p_failed_recall],
            'failed_recall_heaviside': sim.data[self.p_failed_recall_heaviside],
            'pos_recall_state': sim.data[self.p_pos_recall_state],
            'pos_recall_buf': sim.data[self.p_pos_recall_buf],
        }
        # if p.debug:
            # np.savez('debug.npz', **result)
        return result
