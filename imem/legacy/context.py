import nengo
from nengo import spa
import numpy as np

from imem.utils import inhibit_net


def BoundedIntegrator(d, beta, **kwargs):
    kwargs.setdefault('label', "context.BoundedIntegrator")
    net = nengo.Network(**kwargs)

    with net:
        net.input = nengo.Node(size_in=d)

        net.gate = spa.State(d)
        net.current = spa.State(d, feedback=1, neurons_per_dimension=300)
        net.dot = spa.Compare(d)
        with nengo.presets.ThresholdingEnsembles(beta):
            net.update_done = nengo.Ensemble(150, 1)
        net.update_done_th = nengo.Node(size_in=1)

        nengo.Connection(net.input, net.gate.input, synapse=None)
        nengo.Connection(net.input, net.dot.inputA)
        nengo.Connection(net.gate.output, net.current.input, transform=0.3)
        nengo.Connection(net.current.output, net.dot.inputB)

        nengo.Connection(net.dot.output, net.update_done)
        nengo.Connection(
            net.update_done, net.update_done_th, synapse=None,
            function=lambda x: 1 if x >= beta else 0)
        inhibit_net(net.update_done_th, net.gate, strength=3.)

        net.bias_node = nengo.Node(1)
        ctx_square = net.current.state_ensembles.add_output(
            'square', lambda x: x * x)
        with nengo.presets.ThresholdingEnsembles(-0.1):
            net.length = nengo.Ensemble(150, 1)
        nengo.Connection(ctx_square, net.length, transform=-np.ones((1, d)))
        nengo.Connection(net.bias_node, net.length)

        net.downscale = spa.State(d)
        nengo.Connection(net.current.output, net.downscale.input)
        nengo.Connection(
            net.downscale.output, net.current.input, transform=-0.1)
        inhibit_net(net.length, net.downscale, strength=3,
                    function=lambda x: 1 if x >= -0.1 else 0)

        net.output = net.current.output

    return net
