import argparse

from nengo_extras.gexf import CollapsingGexfConverter

from imem.model.imem import IMem, Vocabularies
from imem import protocols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert model to GEXF file for visualiztaion with Gephi.")
    parser.add_argument('filename', type=str, nargs=1, help="output filename")
    args = parser.parse_args()

    proto = protocols.PROTOCOLS['serial']
    stim_provider = protocols.StimulusProvider(proto, distractor_rate=1.)
    vocabs = Vocabularies(stim_provider, 64, 64, proto.n_items)
    model = IMem(stim_provider, vocabs, 0.62676, 0.9775, 0.)

    CollapsingGexfConverter().convert(model).write(args.filename[0])
