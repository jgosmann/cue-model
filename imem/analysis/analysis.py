from collections import defaultdict

import numpy as np
import pandas as pd

from imem.analysis.conversion import convert, register_conversion


def p_first_recall(recalls):
    """Probability of first recall.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos' denoting the recalled
        serial position with an index level 'pos' denoting the output position.

    Returns
    -------
    pandas.DataFrame
    """
    recalls = convert(recalls, 'melted').data
    hist = recalls.xs(0, level='pos').groupby('recalled_pos').size()
    hist /= len(recalls.xs(0, level='pos'))
    hist.name = "Probabilitiy of first recall"
    return hist


def crp(recalls, limit=None):
    """Conditional response probability.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos' denoting the recalled
        serial position with index levels 'trial' identifying the data
        collection trial and 'pos' denoting the output position.
    limit : int
        Limits the lags the CRP is calculated for.

    Returns
    -------
    pandas.DataFrame
    """
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
        for i in range(len(x) - 1):
            row = x.iloc[i]
            pos = int(row['recalled_pos'])
            to_recall.remove(pos)
            for y in to_recall:
                possible_lags['denom'][
                    possible_lags['lag'] == y - pos] += 1
        return possible_lags

    denominator = recalls.reset_index().groupby('trial').apply(get_denom)
    denominator = denominator.reset_index().set_index(['trial', 'lag'])
    if 'level_2' in denominator:
        denominator = denominator.drop(['level_2'], axis=1)
    denominator = np.maximum(denominator, 1)

    crp_data = (numerator['num'] / denominator['denom']).fillna(0).groupby(
        level='lag').mean()
    crp_data[0] = np.nan
    crp_data.name = "Conditional response probability"
    if limit is not None:
        crp_data = crp_data.loc[-limit:limit]
    return crp_data


@register_conversion('melted', 'serial-pos-strict')
def melted_to_serial_pos_strict(data):
    y = defaultdict(lambda: 0, {
        i + 1: (
            x['recalled_pos'] == i).sum() for i, x in data.groupby(level='pos')
    })
    for k in y:
        y[k] /= float(len(data.index.get_level_values('trial').unique()))
    return pd.DataFrame({
        'correct': y
    }, index=y.keys())


@register_conversion('Jahnke68', 'serial-pos-strict')
def jahnke68_to_serial_pos_strict(data):
    return data


@register_conversion('melted', 'serial-pos')
def melted_to_serial_pos(data):
    y = defaultdict(lambda: 0)
    for _, x in data.iterrows():
        if np.isfinite(x['recalled_pos']):
            y[int(x['recalled_pos']) + 1] += 1
    for k in y:
        y[k] /= float(len(data.index.get_level_values('trial').unique()))
    return pd.DataFrame({
        'correct': y
    }, index=y.keys()).sort_index()


def transpositions(data):
    data = convert(data, 'melted').data.dropna()
    x = data['recalled_pos'] - data.index.get_level_values('pos')
    return x.values


def serial_pos_curve(recalls, strict=True):
    """Serial position curve.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos' denoting the recalled
        serial position with index levels 'trial' identifying the data
        collection trial and 'pos' denoting the output position.
    n_items : int
        Number of items presented in the list.

    Returns
    -------
    pandas.DataFrame
    """

    # TODO update documentation + implementation
    fmt = 'serial-pos-strict' if strict else 'serial-pos'
    return convert(recalls, fmt).data
