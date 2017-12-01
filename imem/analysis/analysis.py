from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from imem.analysis.conversion import convert, register_conversion


def bootstrap_ci(data, func, n=3000, p=0.95):
    index = int(n * (1. - p) / 2.)
    r = func(np.random.choice(data, (n, len(data))), axis=1)
    r = np.sort(r)
    return r[index], r[-index]


def aggregate_measure(data, fn):
    return fn(data), bootstrap_ci(data, fn)


def p_first_recall(recalls):
    """Probability of first recall."""
    recalls = convert(recalls, 'melted').data
    hist = recalls.xs(0, level='pos').groupby('recalled_pos').size()
    hist = hist.append(pd.Series(
        {i: 0 for i in range(1, int(max(hist.index)) + 1)
         if i not in hist.index}))
    hist = hist.sort_index()
    n = len(recalls.xs(0, level='pos'))
    ci_low, ci_upp = proportion_confint(hist, n, method='beta')
    hist /= n
    hist = hist.to_frame(name='p_first')
    hist['ci_low'] = hist['p_first'] - ci_low
    hist['ci_upp'] = ci_upp - hist['p_first']
    hist.name = "Probabilitiy of first recall"
    return hist


def crp(recalls):
    """Conditional response probability."""
    recalls = convert(recalls, 'melted').data

    def exclude_repetitions(x):
        values = x.values
        for i, y in enumerate(values):
            if y in values[:i]:
                values[i] = np.nan
        return values
    recalls = recalls.sort_index()
    recalls['recalled_pos'] = recalls['recalled_pos'].groupby(
        level='trial').transform(exclude_repetitions)

    for k in recalls:
        if k != 'recalled_pos':
            recalls.drop([k], axis=1, inplace=True)
    recalls['lag'] = -recalls.groupby(level='trial').diff(-1)

    numerator = pd.DataFrame(
        recalls.reset_index().groupby(['trial', 'lag']).size()).rename(
            columns={0: 'num'})

    n_pos = recalls.index.get_level_values('pos').unique().size

    def get_denom(x):
        to_recall = list(range(n_pos))
        possible_lags = pd.DataFrame({
            'lag': np.arange(-n_pos + 1, n_pos),
            'denom': np.zeros(2 * n_pos - 1)})
        x = x.dropna().sort_values(by='pos')
        for i in range(len(x)):
            row = x.iloc[i]
            pos = int(row['recalled_pos'])
            to_recall.remove(pos)
            for y in to_recall:
                possible_lags['denom'][
                    possible_lags['lag'] == y - pos] += 1
        return possible_lags

    denominator = recalls.reset_index().groupby('trial').apply(get_denom)
    denominator = denominator.reset_index().set_index(['trial', 'lag'])
    for key in ('level_1', 'level_2'):
        if key in denominator:
            denominator = denominator.drop([key], axis=1)
    denominator = np.maximum(denominator, 1)

    df = pd.merge(
        numerator, denominator, left_index=True, right_index=True,
        how='right').fillna(0)
    assert (df['num'] <= df['denom']).all()
    df['crp'] = (df['num'] / df['denom']).fillna(0.)
    crp_data = df.groupby(level='lag').mean()
    crp_data.loc[0, 'crp'] = np.nan
    ci = df['crp'].groupby(level='lag').apply(
        lambda x: pd.Series(bootstrap_ci(x, np.mean)))
    crp_data['ci_low'] = crp_data['crp'] - ci.xs(0, level=1)
    crp_data['ci_upp'] = ci.xs(1, level=1) - crp_data['crp']

    crp_data.name = "Conditional response probability"
    return crp_data


def transpositions(data):
    data = convert(data, 'melted').data.dropna()
    y = data['recalled_pos'] - data.index.get_level_values('pos')
    h, edges = np.histogram(
        y.values, np.arange(min(y.values) - 0.5, max(y.values) + 0.5))
    x = np.asarray(edges[:-1] + 0.5 * np.diff(edges), dtype=int)
    p = h / float(len(y))
    ci_low, ci_upp = proportion_confint(h, len(y), method='beta')
    return pd.DataFrame({
        'p_transpose': p,
        'ci_low': p - ci_low,
        'ci_upp': ci_upp - p,
    }, index=x)


def serial_pos_curve(recalls, strict=True):
    """Serial position curve."""
    fmt = 'serial-pos-strict' if strict else 'serial-pos'
    return convert(recalls, fmt).data


@register_conversion('melted', 'serial-pos-strict')
def melted_to_serial_pos_strict(data):
    y = defaultdict(lambda: 0, {
        i + 1: (
            x['recalled_pos'] == i).sum() for i, x in data.groupby(level='pos')
    })
    y = np.array([y[k] for k in range(1, max(y.keys()) + 1)])
    n = len(data.index.get_level_values('trial').unique())
    ci_low, ci_upp = proportion_confint(y, n, method='beta')
    y = y / float(n)
    return pd.DataFrame({
        'correct': y,
        'ci_low': y - ci_low,
        'ci_upp': ci_upp - y,
    }, index=range(1, len(y) + 1))


@register_conversion('Jahnke68', 'serial-pos-strict')
def jahnke68_to_serial_pos_strict(data):
    return data


@register_conversion('melted', 'serial-pos')
def melted_to_serial_pos(data):
    y = defaultdict(lambda: 0)
    for _, x in data.iterrows():
        if np.isfinite(x['recalled_pos']):
            y[int(x['recalled_pos']) + 1] += 1
    n = len(data.index.get_level_values('trial').unique())
    y = np.array([y[k] for k in range(1, max(y.keys()) + 1)])
    ci_low, ci_upp = proportion_confint(y, n, method='beta')
    m = y / float(n)
    return pd.DataFrame({
        'correct': m,
        'ci_low': m - ci_low,
        'ci_upp': ci_upp - m,
    }, index=np.arange(1, len(y) + 1)).sort_index()


@register_conversion('HowaKaha99', 'melted')
def HowaKaha99_to_melted(data):
    return data


@register_conversion('psyrun', 'psyrun-df')
def psyrun_to_psyrun_df(data):
    d = {
        i: np.asarray(data['responses'], dtype=float)[:, i]
        for i in range(np.asarray(data['responses']).shape[1])}
    d['seed'] = data['seed']
    d['trial'] = data['trial']
    return pd.DataFrame(d)


@register_conversion('psyrun-df', 'melted')
def psyrun_df_to_melted(data):
    return pd.melt(
        pd.DataFrame(data), id_vars=['seed', 'trial'],
        var_name='pos', value_name='recalled_pos').set_index(['trial', 'pos'])


@register_conversion('melted', 'success_count')
def melted_to_success_count(data):
    data = data['recalled_pos']
    return np.squeeze((data >= 0.).groupby(level='trial').sum().values)
