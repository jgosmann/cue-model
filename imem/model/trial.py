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
            model.imem = IMem(
                proto, self.vocabs, p.beta, p.gamma, p.noise,
                p.ose_thr, p.ordinal_prob)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.imem.output_pos, synapse=0.01)

            self.debug_probes = {
                'recall_state': model.imem.tcm.recall.state.output,
                'recall_threshold': model.imem.tcm.recall.threshold.heaviside,
                'recall_buf': model.imem.tcm.recall.buf.output,
                'pos_recall_state': model.imem.tcm.pos_recall.state.output,
                'pos_recall_buf': model.imem.tcm.pos_recall.buf.output,
                'aml_comp': model.imem.tcm.net_m_tf.compare.output,
                'ctx': model.imem.tcm.current_ctx.output,
                'ctx_update': model.imem.tcm.current_ctx.input_update_context,
                'inhib_recall': model.imem.tcm.recall.inhibit.output,
                'recall_ctx': model.imem.tcm.net_m_ft.output,
                'recall_ctx_cue': model.imem.tcm.net_m_ft.input_cue,
                'input_pos': model.imem.tcm.input_pos,
                'current_ctx': model.imem.tcm.current_ctx.output,
                'input_update_ctx':
                    model.imem.tcm.current_ctx.input_update_context,
                'sim_th': model.imem.tcm.sim_th.output,
                'last_item': model.imem.tcm.last_item.output,
                'ose_output': model.imem.ose.output,
                'tcm_output': model.imem.tcm.net_m_tf.output,
                'failed_recall_int': model.imem.tcm.recall.failed_recall_int,
                'failed_recall': model.imem.tcm.recall.failed_recall,
                'failed_recall_heaviside':
                    model.imem.tcm.recall.failed_recall_heaviside,
            }
            if p.debug:
                for k in self.debug_probes:
                    self.debug_probes[k] = nengo.Probe(
                        self.debug_probes[k], synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        proto = self.get_proto(p)
        sim.run(proto.duration + p.recall_duration)

        recall_vocab = self.vocabs.items.create_subset(proto.get_all_items())
        similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        responses = []
        positions = np.arange(p.n_items)
        if proto.serial:
            for i in positions:
                pos_sim = self.vocabs.positions["P" + str(i)].dot(
                    sim.data[self.p_pos].T)
                recall_phase = sim.trange() > proto.pres_phase_duration
                t = sim.trange()[recall_phase & (pos_sim > 0.8)]
                recall_for_pos = np.mean(similarity[t], axis=0)
                if np.any(recall_for_pos > 0.8):
                    responses.append(float(np.argmax(recall_for_pos)))
                else:
                    responses.append(np.nan)
        else:
            above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
            for x in np.argmax(above_threshold, axis=1):
                if x not in responses:
                    responses.append(float(x))
        responses = responses + (p.n_items - len(responses)) * [np.nan]

        result = {
            'responses': responses,
            'positions': positions,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
