"""Analysis tools for context networks."""

import itertools

from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt
from nengo import spa
import numpy as np
import seaborn as sns


def context_vectors(vocab, beta, init='INIT', stim_prefix='V'):
    v = vocab.parse(init).v
    yield v
    rho = np.sqrt(1. - beta**2)
    for s in stimulus_vectors(vocab, beta, prefix=stim_prefix):
        v = rho * v + beta * s
        v /= np.linalg.norm(v)
        yield v


def stimulus_vectors(vocab, beta, prefix='V'):
    """Generator for stimulus vectors.

    Starts with V0 and in each step thereafter the generated vector will be
    sqrt(1 - beta**2) * V[i-1] + beta * V[i].

    Parameters
    ----------
    vocab : spa.Vocabulary
        Vocabulary to generate the stimulus vectors.
    beta : float
        Strength of newly added in stimulus vector.
    """
    v = vocab.parse('V0').v
    yield v
    for i in itertools.count(1):
        v = np.sqrt(1. - beta**2) * v + beta *  vocab.parse('V' + str(i)).v
        v /= np.linalg.norm(v)
        yield v

class ContextTestEnv(object):
    """Environment for testing of context network.

    Parameters
    ----------
    seed : int
        Seed for vocabulary.
    d : int
        Dimensionality of vocabulary.
    beta : float
        TCM beta parameter to use for test.
    beta_stim : float
        Dissimilarity between successive stimulus vectors.
    n : int
        Number of stimulus vectors to present.
    dt : float
        Simulation time step.

    Attributes
    ----------
    seed : int
        Seed for vocabulary.
    d : int
        Dimensionality of vocabulary.
    beta : float
        TCM beta parameter to use for test.
    n : int
        Number of stimulus vectors to generate and present.
    dt : float
        Simulation time step.
    init_phase : float
        Initialization phase for the context network
    vocab : Vocabulary
        The SPA vocabulary to produce stimulus vectors and the initial context
        vector.
    recalled_ctxs : list
        The recalled context vectors to provide to the context network.
    """
    def __init__(self, seed, d=64, beta=0.6, beta_stim=1., n=10, dt=0.001,
                 stimulus_gen=None):
        self.seed = seed
        self.d = d
        self.beta = beta
        self.n = n
        self.dt = dt

        self.init_phase = 0.5

        self.vocab = spa.Vocabulary(d, rng=np.random.RandomState(seed=seed))
        self.recalled_ctxs = [
            x for i, x in zip(range(n), stimulus_vectors(
                self.vocab, beta_stim))]

    def recalled_ctx_fn(self, t):
        """Function returning the recalled context at time *t*.

        Will present the null vector for the first second and then one
        vector per second.
        """
        if t < 1.:
            return np.zeros(self.d)
        else:
            return self.recalled_ctxs[min(int(t), len(self.recalled_ctxs) - 1)]

    def context_init_fn(self, t, x):
        """Context initialization function.

        Returns the difference of the initial context vector and *x*
        for *init_phase* seconds.
        """
        if t < self.init_phase:
            return self.vocab.parse('InitCtx').v - x
        else:
            return np.zeros(self.d)


def plot_ctx_net_analysis(t, recalled_ctx, ctx, ctx_test_env, fig=None):
    """Plot complete context network analysis.
    Parameters
    ----------
    t : ndarray
        Time values.
    recalled_ctx : ndarray
        Recalled context vectors.
    ctx : ndarray
        Context vectors.
    ctx_test_env : `ContextTestEnv`
        Context test environment.
    fig : Figure
        Figure to plot on.
    """
    if fig is None:
        fig = plt.gcf()

    subplots = gridspec.GridSpec(2, 2, width_ratios=(1., 2.))
    ax1 = fig.add_subplot(subplots[0, 0])
    ax1.set_title("(a) Norm")
    ax2 = fig.add_subplot(subplots[1, 0], sharex=ax1)
    ax2.set_title("(b) Effective context drift")
    ax3 = fig.add_subplot(subplots[:, 1], sharex=ax1)
    ax3.set_title("(c) Context similarity decay")

    plot_ctx_norm(t, ctx, ax=ax1)
    plot_effective_beta(t, recalled_ctx, ctx, ctx_test_env, ax=ax2)
    plot_similarity_decay(t, ctx, ctx_test_env, ax=ax3)

    sns.despine(fig)
    subplots.tight_layout(fig)

def plot_ctx_norm(t, ctx, ax=None):
    """Plot the context norm over time.

    Parameters
    ----------
    t : ndarray
        Time values.
    ctx : ndarray
        Context vectors.
    ax : Axes
        Axes to plot on.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(t, np.linalg.norm(ctx, axis=1))
    ax.axhline(y=1., c='k', ls='--')
    ax.set_ylabel("$||\mathbf{c}||$")
    ax.set_yticks([0, 1])


def plot_effective_beta(t, recalled_ctx, ctx, ctx_test_env, ax=None):
    """Plot the effective beta over time.

    Parameters
    ----------
    t : ndarray
        Time values.
    recalled_ctx : ndarray
        Recalled context vectors.
    ctx : ndarray
        Context vectors.
    ctx_test_env : `ContextTestEnv`
        Context test environment.
    ax : Axes
        Axes to plot on.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_prop_cycle('color', sns.color_palette("husl", ctx_test_env.n))
    y = np.sum(recalled_ctx * ctx, axis=1)
    for i in range(1, ctx_test_env.n):
        sel = (t > i) & (t <= i + 1)
        ax.plot(t[sel], y[sel])

    ax.axhline(y=ctx_test_env.beta, c='k', ls='--')
    ax.set_xlabel(r"Time $t/\mathrm{s}$")
    ax.set_ylabel(r"$\beta'$")
    ax.set_yticks([0, ctx_test_env.beta, 1])


def plot_similarity_decay(
        t, ctx, ctx_test_env, max_diff=0.2, ax=None):
    """Plot the similarity decay over time.

    Parameters
    ----------
    t : ndarray
        Time values.
    ctx : ndarray
        Context vectors.
    ctx_test_env : `ContextTestEnv`
        Context test environment.
    max_diff : float
        Maximum difference to target beta to highlight in plot.
    ax : Axes
        Axes to plot on.
    """
    if ax is None:
        ax = plt.gca()

    with sns.color_palette("GnBu_d", ctx_test_env.n):
        out_normed = ctx / np.linalg.norm(ctx, axis=1)[:, None]
        for i in range(1, ctx_test_env.n):
            start = int((i + .7) / ctx_test_env.dt)
            end = int((i + 1.) / ctx_test_env.dt)
            target = np.mean(out_normed[start:end], axis=0)
            y = np.dot(out_normed, target)
            ax.plot(t - i, y, c=sns.color_palette()[-i])

        decay = lambda x: np.sqrt(1. - x**2)**np.floor(t)
        ax.plot(t, decay(ctx_test_env.beta), color='gray')

        ax.set_xlim(left=0.)
        ax.set_ylim(0., 1.)
        ax.set_xlabel(r"Time $t/\mathrm{s}$")
        ax.set_ylabel(r"$\mathbf{c}_i \cdot \mathbf{c}(t)$")

    return ax

def band_average(mat):
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    in_size = mat.shape[0]
    out_size = 2 * in_size - 1

    avg = np.zeros(out_size)
    for i in range(in_size):
        start = in_size - i - 1
        end = start + in_size
        avg[start:end] += mat[i, :]
    avg /= in_size - np.abs(in_size - 1 - np.arange(out_size))
    return avg
