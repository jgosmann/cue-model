from glob import glob
import os.path
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psyrun.store import AutodetectStore
import seaborn as sns

from imem.analysis import analysis
from imem.analysis.conversion import convert, DataRep, register_conversion
from imem.analysis.io import read_exp_data
from imem.protocols import PROTOCOLS


store = AutodetectStore()


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


def evaluate(path):
    for proto_name, proto in PROTOCOLS.items():
        try:
            proto_path = os.path.join(path, proto_name)
            if os.path.exists(proto_path):
                exp_data = read_exp_data(proto.exp_data)
                model_data = DataRep(
                    'psyrun', store.load(locate_results_file(proto_path)))

                if proto.serial:
                    fig = plt.figure(figsize=(12, 4))
                    evaluate_serial_recall(proto, exp_data, model_data, fig=fig)
                else:
                    fig = plt.figure(figsize=(12, 8))
                    evaluate_free_recall(proto, exp_data, model_data, fig=fig)

                fig.suptitle(path + ', ' + proto_name)
                fig.tight_layout(rect=(.0, .0, 1., .95))
        except Exception as err:
            traceback.print_exc()
            warnings.warn(str(err))


def evaluate_serial_recall(proto, exp_data, model_data, fig=None):
    if fig is None:
        fig = plt.gcf()

    evaluate_serial_pos_curve(
        proto, exp_data, model_data, ax=fig.add_subplot(1, 2, 1))
    evaluate_transpositions(
        proto, exp_data, model_data, ax=fig.add_subplot(1, 2, 2))


def evaluate_free_recall(proto, exp_data, model_data, fig=None):
    if fig is None:
        fig = plt.gcf()

    evaluate_successful_recalls(
        proto, exp_data, model_data, ax=fig.add_subplot(2, 2, 1))
    evaluate_p_first_recall(
        proto, exp_data, model_data, ax=fig.add_subplot(2, 2, 2))
    evaluate_crp(
        proto, exp_data, model_data, ax=fig.add_subplot(2, 2, 3))
    evaluate_serial_pos_curve(
        proto, exp_data, model_data, strict=False, ax=fig.add_subplot(2, 2, 4))


def evaluate_successful_recalls(proto, exp_data, model_data, ax=None):
    cp = iter(sns.color_palette())

    if ax is None:
        ax = plt.gca()

    plot_successful_recalls(
        exp_data, proto.n_items, color=next(cp),
        label="experimental", ax=ax)
    plot_successful_recalls(
        model_data, proto.n_items, color=next(cp),
        label="model", ax=ax)

    ax.set_xlim(-0.5, proto.n_items + 0.5)
    ax.set_xlabel("# successful recalls")
    ax.set_ylabel("Proportion")
    ax.legend()


def evaluate_p_first_recall(proto, exp_data, model_data, ax=None):
    if ax is None:
        ax = plt.gca()

    analysis.p_first_recall(exp_data).plot(
        marker='o', label="experimental", ax=ax)
    analysis.p_first_recall(model_data).plot(
        marker='o', label="model", ax=ax)

    ax.set_xlabel("Serial position")
    ax.set_ylabel("Probability of first recall")
    ax.legend()


def evaluate_crp(proto, exp_data, model_data, ax=None, limit=6):
    if ax is None:
        ax = plt.gca()

    analysis.crp(exp_data, limit=limit).plot(
        marker='o', label="experimental", ax=ax)
    analysis.crp(model_data, limit=limit).plot(
        marker='o', label="model", ax=ax)

    ax.set_xlabel("Lag position")
    ax.set_ylabel("CRP")
    ax.legend()


def evaluate_serial_pos_curve(
        proto, exp_data, model_data, strict=True, ax=None):
    if ax is None:
        ax = plt.gca()

    analysis.serial_pos_curve(exp_data, strict=strict).plot(
        marker='o', label="experimental", ax=ax)
    analysis.serial_pos_curve(model_data, strict=strict).plot(
        marker='o', label="model", ax=ax)

    ax.set_xlabel("Serial position")
    ax.set_ylabel("Proportion correct recalls")
    ax.set_ylim(0, 1)
    ax.legend()


def evaluate_transpositions(proto, exp_data, model_data, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.hist(analysis.transpositions(model_data), bins=13, range=(-6.5, 6.5))


def plot_successful_recalls(
        n_successfull, n_items, ax=None, label=None, **kwargs):
    n_successfull = convert(n_successfull, 'success_count').data
    if ax is None:
        ax = plt.gca()
    ax.hist(
        n_successfull, bins=n_items + 1, range=(-0.5, n_items + 0.5),
        density=True, alpha=0.5, label=label, **kwargs)
    if label is not None:
        label = label + ' (mean)'
    ax.axvline(x=np.mean(n_successfull.data), label=label, **kwargs)


def locate_results_file(path):
    candidates = glob(os.path.join(path, 'result.*'))
    if len(candidates) < 1:
        raise FileNotFoundError("No results file found in {!r}.".format(path))
    elif len(candidates) > 1:
        warnings.warn(
            "Found multiple results file in {!r}, using {!r}.".format(
                path, candidates[0]))
    return candidates[0]
