import os.path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from imem.analysis.conversion import DataRep


def next_line_or_stop(f):
    line = f.readline()
    if not line:
        raise StopIteration()
    return line


class HowaKaha99FrameReader(object):
    def __init__(self, filename, n_items=12):
        self.filename = filename
        self.n_items = n_items
        self.trial = 0
        self._fd = None

    def __enter__(self):
        self._fd = open(self.filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._fd.close()

    def read_frame(self):
        subject = next_line_or_stop(self._fd).split()[0]
        next_line_or_stop(self._fd)
        responses = np.empty(self.n_items)
        responses.fill(np.nan)
        for i, pos in enumerate(next_line_or_stop(self._fd).split()):
            pos = int(pos)
            if pos > 0:
                responses[i] = pos
        for i in range(3):
            line = next_line_or_stop(self._fd)
            while i < 2 and line.strip() == '':
                line = next_line_or_stop(self._fd)
        assert line.strip() == '', line

        df = pd.DataFrame({
            'trial': self.trial,
            'subject': subject,
            'pos': np.arange(self.n_items),
            'recalled_pos': responses - 1
        }).set_index(['subject', 'trial', 'pos'])
        self.trial += 1
        return df


def read_HowaKaha99(filename):
    responses = []
    with HowaKaha99FrameReader(filename) as fr:
        responses = []
        try:
            while True:
                responses.append(fr.read_frame())
        except StopIteration:
            pass
    return pd.concat(responses)


def read_Jahnke68(filename):
    data = pd.read_csv(filename, header=0, names=['pos', 'correct'])
    data['pos'] = data['pos'].round().astype(int)
    data = data.set_index('pos')
    n = 96
    ci_low, ci_upp = proportion_confint(data['correct'] * n, n, method='beta')
    data['ci_low'] = data['correct'] - ci_low
    data['ci_upp'] = ci_upp - data['correct']
    return data


def read_exp_data(path):
    dataformat = os.path.basename(os.path.dirname(path))
    return DataRep(dataformat, globals()['read_' + dataformat](path))
