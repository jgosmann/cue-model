import nengo
import numpy as np


def inhibit_net(pre, post, strength=2., **kwargs):
    """Makes an inhibitory connection to all ensembles of a network.

    Uses a lowpass synapse with a time constant of 0.0085s by default.

    Parameters
    ----------
    pre : 1-d nengo.Node or nengo.Ensemble
        Scalar source of the inhibition.
    post : nengo.Network
        Target of the inhibition.
    strength : float
        Strength of the inhibition.
    kwargs : dict
        Additional keyword arguments for the created connections.

    Returns
    -------
    list
        Created connections.
    """

    kwargs.setdefault('synapse', 0.0085)
    return [nengo.Connection(
        pre, e.neurons, transform=-strength * np.ones((e.n_neurons, 1)),
        **kwargs) for e in post.all_ensembles]
