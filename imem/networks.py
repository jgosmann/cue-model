import nengo
import numpy as np


def OneHotCounter(n, **kwargs):
    with nengo.Network(**kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(0.):
            net.state = nengo.networks.EnsembleArray(20, n)
            net.rising_edge_detector = nengo.Ensemble(50, 1)

        net.bias = nengo.Node(1.)
        nengo.Connection(net.bias, net.state.input,
                         transform=-0.3 * np.ones((n, 1)))

        nengo.Connection(
            net.state.output, net.state.input, synapse=0.1, transform=2.)
        nengo.Connection(
            net.state.output, net.state.input,
            transform=0.2 * np.roll(np.eye(n), 1, axis=1))
        nengo.Connection(
            net.state.output, net.state.input,
            transform=-np.roll(np.eye(n), -1, axis=1))

        net.input_inc = nengo.Node(size_in=1)
        nengo.Connection(net.input_inc, net.rising_edge_detector, synapse=0.05,
                         transform=-1)
        nengo.Connection(net.input_inc, net.rising_edge_detector,
                         synapse=0.005, transform=1)
        nengo.Connection(net.rising_edge_detector, net.state.input,
                         transform=0.6 * np.ones((n, 1)))

        net.input = net.state.input
        net.output = net.state.output

    return net
