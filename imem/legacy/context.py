"""Legacy context networks that did not work as part of the TCM."""

import nengo
import nengo_spa as spa
import numpy as np

from imem.modules import GatedMemory
from imem.utils.nengo import inhibit_net


def BoundedIntegrator(d, beta, **kwargs):
    kwargs.setdefault('label', "context.BoundedIntegrator")
    net = nengo.Network(**kwargs)

    with net:
        net.input = nengo.Node(size_in=d)

        net.bias_node = nengo.Node(1)
        net.gate = spa.State(d)
        net.current = spa.State(d, feedback=1, neurons_per_dimension=300)
        net.dot = spa.Compare(d)
        with nengo.presets.ThresholdingEnsembles(0.):
            net.update_done = nengo.Ensemble(150, 1)
        net.update_done_th = nengo.Node(size_in=1)

        nengo.Connection(net.input, net.gate.input, synapse=None)
        nengo.Connection(net.input, net.dot.input_a)
        nengo.Connection(net.gate.output, net.current.input, transform=0.3)
        nengo.Connection(net.current.output, net.dot.input_b)

        nengo.Connection(net.dot.output, net.update_done)
        nengo.Connection(net.bias_node, net.update_done, transform=-beta)
        nengo.Connection(
            net.update_done, net.update_done_th, synapse=None,
            function=lambda x: x > 0)
        inhibit_net(net.update_done_th, net.gate, strength=3.)

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


def AlternatingMemoryBuffers(d, beta, **kwargs):
    kwargs.setdefault('label', "context.AlternatingMemories")

    with nengo.Network(**kwargs) as net:
        net.input = nengo.Node(size_in=d)
        net.new_ctx = nengo.Node(size_in=d)
        net.current = GatedMemory(d)
        net.old = GatedMemory(d)
        net.dot = spa.Compare(d)
        net.bias = nengo.Node(1.)
        with nengo.presets.ThresholdingEnsembles(0.):
            net.update_done = nengo.Ensemble(150, 1)
        net.update_done_th = nengo.Node(size_in=1)

        nengo.Connection(net.bias, net.update_done, transform=-1.1)

        nengo.Connection(net.input, net.new_ctx, transform=beta)
        nengo.Connection(net.new_ctx, net.current.input, synapse=None)

        nengo.Connection(
            net.update_done, net.update_done_th, synapse=None,
            function=lambda x: x > 0)

        nengo.Connection(net.new_ctx, net.dot.input_a)
        nengo.Connection(net.current.output, net.dot.input_b)
        nengo.Connection(net.dot.output, net.update_done)
        nengo.Connection(net.update_done_th, net.current.store)

        nengo.Connection(net.current.output, net.old.input)
        nengo.Connection(
            net.old.output, net.new_ctx, transform=np.sqrt(1. - (beta)**2))
        with nengo.presets.ThresholdingEnsembles(0.):
            net.invert = nengo.Ensemble(50, 1)
        nengo.Connection(net.bias, net.invert)
        nengo.Connection(
            net.update_done_th, net.invert.neurons,
            transform=-3 * np.ones((net.invert.n_neurons, 1)))
        nengo.Connection(net.invert, net.old.store)

        net.output = net.current.output

    return net


def ThreeMemory(d, beta, **kwargs):
    kwargs.setdefault('label', "context.ThreeMemory")

    with nengo.Network(**kwargs) as net:
        net.input = nengo.Node(size_in=d)

        net.new_ctx = nengo.Node(size_in=d)
        net.current = GatedMemory(d)
        net.buf = GatedMemory(d)
        net.old = GatedMemory(d)

        nengo.Connection(net.input, net.new_ctx, transform=beta)
        nengo.Connection(
            net.old.mem.output, net.new_ctx, transform=np.sqrt(1. - (beta)**2))
        nengo.Connection(net.new_ctx, net.current.diff.input)
        nengo.Connection(net.current.mem.output, net.buf.diff.input)
        nengo.Connection(net.buf.mem.output, net.old.diff.input)

        net.bias = nengo.Node(1)
        net.dot = spa.Compare(d)
        with nengo.presets.ThresholdingEnsembles(0.0):
            net.update_done = nengo.Ensemble(150, 1)
            nengo.Connection(net.bias, net.update_done, transform=-beta + 0.1)
        net.update_done_th = nengo.Node(size_in=1)
        nengo.Connection(net.input, net.dot.input_a)
        nengo.Connection(net.current.mem.output, net.dot.input_b)
        nengo.Connection(net.dot.output, net.update_done)
        nengo.Connection(
            net.update_done, net.update_done_th, synapse=None,
            function=lambda x: 1 if x >= 0.0 else 0)

        nengo.Connection(net.update_done_th, net.current.store)
        nengo.Connection(net.update_done_th, net.old.store)
        nengo.Connection(net.bias, net.buf.store)
        nengo.Connection(net.update_done_th, net.buf.store, transform=-1)

        net.output = net.current.mem.output

    return net


def Context4(d, beta, **kwargs):
    with nengo.Network(**kwargs) as net:
        net.input = nengo.Node(size_in=d)

        net.new_ctx = nengo.Node(size_in=d)
        net.current = GatedMemory(d)
        net.buf = GatedMemory(d)
        net.old = GatedMemory(d)

        nengo.Connection(net.input, net.new_ctx, transform=beta)
        nengo.Connection(
            net.old.mem.output, net.new_ctx,
            transform=np.sqrt(1. - (beta)**2))
        nengo.Connection(net.new_ctx, net.current.diff.input)
        nengo.Connection(net.current.mem.output, net.buf.diff.input)
        nengo.Connection(net.buf.mem.output, net.old.diff.input)

        net.bias = nengo.Node(1)
        net.input_update_context = nengo.Node(size_in=1)

        nengo.Connection(net.bias, net.current.store)
        nengo.Connection(net.bias, net.old.store)
        nengo.Connection(
            net.input_update_context, net.current.store, transform=-1.,
            synapse=None)
        nengo.Connection(
            net.input_update_context, net.old.store, transform=-1.,
            synapse=None)

        nengo.Connection(net.input_update_context, net.buf.store)

        # net.downscale = spa.State(d)
        # nengo.Connection(net.buf.mem.output, net.downscale.input)
        # nengo.Connection(
            # net.downscale.output, net.buf.mem.input, transform=-.1)
        # nengo.Connection(
            # net.downscale.output, net.buf.diff.input, transform=-.1)
        # sq = net.buf.mem.state_ensembles.add_output('sq', np.square)
        # with nengo.presets.ThresholdingEnsembles(0.):
            # net.downscale_activate = nengo.Ensemble(50, 1)
        # nengo.Connection(net.bias, net.downscale_activate)
        # nengo.Connection(
            # sq, net.downscale_activate, transform=-np.ones((1, d)))
        # inhibit_net(net.downscale_activate, net.downscale.state_ensembles,
                    # strength=3)

        net.output = net.current.mem.output

    return net


def Context5(d, beta, **kwargs):
    with nengo.Network(**kwargs) as net:
        net.input = nengo.Node(size_in=d)

        net.new_ctx = nengo.Node(size_in=d)
        net.current = GatedMemory(d)
        net.old = GatedMemory(d)

        nengo.Connection(net.input, net.new_ctx, transform=beta)
        nengo.Connection(
            net.old.mem.output, net.new_ctx,
            transform=np.sqrt(1. - (beta)**2))
        nengo.Connection(net.new_ctx, net.current.diff.input)
        nengo.Connection(net.current.mem.output, net.old.diff.input)

        net.bias = nengo.Node(1)
        net.input_update_context = nengo.Node(size_in=1)

        nengo.Connection(net.bias, net.current.store)
        nengo.Connection(
            net.input_update_context, net.current.store, transform=-1.,
            synapse=None)

        nengo.Connection(net.input_update_context, net.old.store)

        # net.downscale = spa.State(d)
        # nengo.Connection(net.buf.mem.output, net.downscale.input)
        # nengo.Connection(
            # net.downscale.output, net.buf.mem.input, transform=-.1)
        # nengo.Connection(
            # net.downscale.output, net.buf.diff.input, transform=-.1)
        # sq = net.buf.mem.state_ensembles.add_output('sq', np.square)
        # with nengo.presets.ThresholdingEnsembles(0.):
            # net.downscale_activate = nengo.Ensemble(50, 1)
        # nengo.Connection(net.bias, net.downscale_activate)
        # nengo.Connection(
            # sq, net.downscale_activate, transform=-np.ones((1, d)))
        # inhibit_net(net.downscale_activate, net.downscale.state_ensembles,
                    # strength=3)

        net.output = net.current.mem.output

    return net
