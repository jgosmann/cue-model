from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem.analysis.neural import model_out_to_responses
from imem.model import IMem, Vocabularies
from imem.protocols import PROTOCOLS, StimulusProvider


class IMemTrial(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    def params(self):
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=.4)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.1)
        self.param("TCM prob. to recall from beginning", ordinal_prob=.1)
        self.param("noise in recall", noise=0.009)
        self.param("min. recall evidence", min_evidence=0.035)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        self.proto = PROTOCOLS[p.protocol]
        self.stim_provider = StimulusProvider(self.proto, p.distractor_rate)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d, self.proto.n_items + 3,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.imem = IMem(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.imem.output_pos, synapse=0.01)

            self.debug_probes = {
                'recall_state': model.imem.recall.state.output,
                'recall_threshold': model.imem.recall.threshold.heaviside,
                'recall_buf': model.imem.recall.buf.output,
                'pos_recall_state': model.imem.pos_recall.state.output,
                'pos_recall_buf': model.imem.pos_recall.buf.output,
                'aml_comp': model.imem.tcm.net_m_tf.compare.output,
                'ctx': model.imem.tcm.current_ctx.output,
                'ctx_update': model.imem.tcm.current_ctx.input_update_context,
                'inhib_recall': model.imem.recall.inhibit.output,
                'out_inhibit_gate': model.imem.recall.out_inhibit_gate.output,
                'recall_ctx': model.imem.tcm.net_m_ft.output,
                'recall_ctx_cue': model.imem.tcm.net_m_ft.input_cue,
                'input_pos': model.imem.tcm.input_pos,
                'current_ctx': model.imem.tcm.current_ctx.output,
                'input_update_ctx': model.imem.tcm.input_update_context,
                'sim_th': model.imem.sim_th.output,
                'last_item': model.imem.last_item.output,
                'ose_output': model.imem.ose.output,
                'tcm_output': model.imem.tcm.net_m_tf.output,
                'failed_recall_int': model.imem.recall.failed_recall_int,
                'failed_recall': model.imem.recall.failed_recall,
                'failed_recall_heaviside':
                    model.imem.recall.failed_recall_heaviside,
                'start_of_serial_recall': model.imem.start_of_serial_recall,
                'start_of_free_recall': model.imem.start_of_free_recall,
                'pos_state': model.imem.pos.state.output,
                'pos_state_in': model.imem.pos.state.input,
                'pos_inhibit_threshold': model.imem.pos.inhibit_threshold.output,
                'pos_advance_threshold': model.imem.pos.advance_threshold.output,
                'input_inc': model.imem.pos.input_inc,
                'no_pos_count': model.imem.ctrl.output_no_pos_count,
                'ose_recall_gate': model.imem.ose_recall_gate.output,
                'tcm_recall_gate': model.imem.tcm_recall_gate.output,
                'pos_gate': model.imem.pos_gate.output,
                'ose': model.imem.ose.mem.output,
                'buf_input_store': model.imem.recall.buf_input_store,
                # 'out_inhib_gate_update': model.imem.recall.out_inhib_gate_update,
                # 'input_update_inhibit': model.imem.recall.input_update_inhibit,
                'sim_th_neg': model.imem.sim_th_neg,
                'sim_th_pos': model.imem.sim_th_pos,
            }
            if p.debug:
                for k in self.debug_probes:
                    self.debug_probes[k] = nengo.Probe(
                        self.debug_probes[k], synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.proto.duration + p.recall_duration)

        recall_vocab = self.vocabs.items.create_subset(self.stim_provider.get_all_items())
        responses = model_out_to_responses(
            recall_vocab, sim.trange(), sim.data[self.p_recalls],
            sim.data[self.p_pos], self.proto)

        result = {
            'responses': responses,
            'pos': sim.data[self.p_pos],
            'recalls': sim.data[self.p_recalls],
            'positions': np.arange(self.proto.n_items),
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
