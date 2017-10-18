import argparse

from nengo_extras.gexf import CollapsingGexfConverter

from imem.model.imem import IMem, Vocabularies
from imem import protocols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert model to GEXF file for visualiztaion with Gephi.")
    parser.add_argument('filename', type=str, nargs=1, help="output filename")
    args = parser.parse_args()

    proto = protocols.Recall(
        pi=1., ipi=0., ri=0., serial=True, n_items=15, distractor_rate=1.)
    vocabs = Vocabularies(proto, 64, 64, proto.n_items)
    model = IMem(proto, vocabs, 0.62676, 0.9775, 0.)

    CollapsingGexfConverter().convert(model).write(args.filename[0])
