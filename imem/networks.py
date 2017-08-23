import nengo
import numpy as np


# def OneHotCounter(n, **kwargs):
    # with nengo.Network(**kwargs) as net:
        # with nengo.presets.ThresholdingEnsembles(0.):
            # net.state = nengo.networks.EnsembleArray(20, n)
            # net.rising_edge_detector = nengo.Ensemble(50, 1)

        # net.bias = nengo.Node(1.)
        # nengo.Connection(net.bias, net.state.input,
                         # transform=-0.6 * np.ones((n, 1)))

        # nengo.Connection(
            # net.state.output, net.state.input, synapse=0.1, transform=2.)
        # nengo.Connection(
            # net.state.output, net.state.input,
            # transform=0.4 * np.roll(np.eye(n), -1, axis=1))
        # nengo.Connection(
            # net.state.output, net.state.input,
            # transform=-np.roll(np.eye(n), 1, axis=1))

        # with nengo.presets.ThresholdingEnsembles(0.):
            # net.rising_edge_gate = nengo.Ensemble(50, 1)
        # net.input_inc = nengo.Node(size_in=1)
        # nengo.Connection(net.input_inc, net.rising_edge_detector, synapse=0.05,
                         # transform=-1)
        # nengo.Connection(net.input_inc, net.rising_edge_detector,
                         # synapse=0.005, transform=1)
        # nengo.Connection(net.rising_edge_detector, net.rising_edge_gate)
        # nengo.Connection(net.rising_edge_gate, net.state.input,
                         # transform=0.8 * np.ones((n, 1)))


        # net.input = net.state.input
        # net.output = net.state.output

    # return net

def OneHotCounter(n, **kwargs):
    with nengo.Network(seed=73, **kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(0.):
            net.state = nengo.networks.EnsembleArray(40, n)
            net.inhibit_threshold = nengo.networks.EnsembleArray(40, n)
            net.advance_threshold = nengo.networks.EnsembleArray(40, n)
            net.rising_edge_detector = nengo.Ensemble(50, 1)
            net.rising_edge_gate = nengo.Ensemble(50, 1)

        net.bias = nengo.Node(1.)

        net.input_inc = nengo.Node(size_in=1)
        nengo.Connection(net.input_inc, net.rising_edge_detector, synapse=0.05,
                         transform=-1)
        nengo.Connection(net.input_inc, net.rising_edge_detector,
                         synapse=0.005, transform=1)
        nengo.Connection(net.rising_edge_detector, net.rising_edge_gate)

        nengo.Connection(
            net.bias, net.state.input, transform=-0.2 * np.ones((n, 1)))
        nengo.Connection(
            net.state.add_output('const', lambda x: 1.2 if x > 0. else 0.), net.state.input,
            synapse=0.1)

        nengo.Connection(
            net.bias, net.inhibit_threshold.input,
            transform=-0.6 * np.ones((n, 1)))
        nengo.Connection(net.state.const, net.inhibit_threshold.input)
        nengo.Connection(
            net.inhibit_threshold.add_output('heaviside', lambda x: x > 0.),
            net.state.input, transform=-2. * np.roll(np.eye(n), 1, axis=1), synapse=0.1)
        nengo.Connection(
            net.inhibit_threshold.heaviside,
            net.state.input, transform=-2. * np.roll(np.eye(n), -2, axis=1), synapse=0.1)

        nengo.Connection(
            net.state.output, net.advance_threshold.input,
            transform=-(np.ones((n, n)) - np.eye(n)))
        nengo.Connection(net.bias, net.rising_edge_gate, transform=-0.1)
        nengo.Connection(
            net.rising_edge_gate, net.advance_threshold.input,
            transform=0.8 * np.ones((n, 1)), function=lambda x: x > 0)
        nengo.Connection(
            net.advance_threshold.output, net.state.input,
            transform=2. * np.roll(np.eye(n), -1, axis=1), synapse=0.1)

        net.input = net.state.input
        net.output = net.state.const

    return net
