"""Convert neural output to formats more readily analyzable."""

import nengo_spa as spa
import numpy as np


def model_out_to_responses(recall_vocab, t, model_out, pos_out, proto):
    similarity = spa.similarity(model_out, recall_vocab)
    responses = []
    positions = np.arange(proto.n_items)
    last_recall = -1
    if proto.serial:
        for i in positions:
            recall_phase = t > proto.pres_phase_duration
            s = recall_phase & (pos_out[:, i] > 0.8)
            if np.any(s):
                recall_for_pos = similarity[s][-1, :]
            else:
                recall_for_pos = np.array([0.])
            if np.any(recall_for_pos > 0.6):
                recalled = float(np.argmax(recall_for_pos))
                if len(responses) == 0 or recalled != last_recall:
                    responses.append(recalled)
                    last_recall = recalled
                else:
                    responses.append(np.nan)
            else:
                responses.append(np.nan)
    else:
        above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
        for x in np.argmax(above_threshold, axis=1):
            if x not in responses:
                responses.append(float(x))
    responses = responses + (proto.n_items - len(responses)) * [np.nan]
    return responses
