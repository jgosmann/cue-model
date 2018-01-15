from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem.model import IMem, Vocabularies
from imem.protocols import HebbRepStimulusProvider


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

    def model(self, p):
        self.stim_provider = HebbRepStimulusProvider(
            n_total_items = 19,
            n_items_per_list = 7,
            n_lists = 2 * 4,
            rep_list_freq = 2,
            pi = 1.,
            recall_duration = 40.)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d,
            self.stim_provider.n_total_items,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.imem = IMem(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.imem.output_pos, synapse=0.01)

            self.debug_probes = {
                'start_of_serial_recall': model.imem.start_of_serial_recall,
                'start_of_free_recall': model.imem.start_of_free_recall,
                'start_of_pres_phase': model.imem.start_of_pres_phase,
                'no_learn': model.imem.ctrl.output_no_learn,
            }
            if p.debug:
                for k in self.debug_probes:
                    self.debug_probes[k] = nengo.Probe(
                        self.debug_probes[k], synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        # sim.run(self.proto.total_duration)
        sim.run(60.)

        # recall_vocab = self.vocabs.items.create_subset(self.stim_provider.get_all_items())
        # similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        # responses = []
        # positions = np.arange(self.proto.n_items)
        # last_recall = -1
        # if self.proto.serial:
            # for i in positions:
                # recall_phase = sim.trange() > self.proto.pres_phase_duration
                # s = recall_phase & (sim.data[self.p_pos][:, i] > 0.8)
                # if np.any(s):
                    # recall_for_pos = similarity[s][-1, :]
                # else:
                    # recall_for_pos = np.array([0.])
                # if np.any(recall_for_pos > 0.6):
                    # recalled = float(np.argmax(recall_for_pos))
                    # if len(responses) == 0 or recalled != last_recall:
                        # responses.append(recalled)
                        # last_recall = recalled
                    # else:
                        # responses.append(np.nan)
                # else:
                    # responses.append(np.nan)
        # else:
            # above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
            # for x in np.argmax(above_threshold, axis=1):
                # if x not in responses:
                    # responses.append(float(x))
        # responses = responses + (self.proto.n_items - len(responses)) * [np.nan]

        result = {
            # 'responses': responses,
            'pos': sim.data[self.p_pos],
            'recalls': sim.data[self.p_recalls],
            # 'positions': positions,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
            'lists': self.proto.lists,
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
