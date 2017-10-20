from collections import namedtuple
import os.path

import numpy as np


class Recall(namedtuple(
        'Recall', ['n_items', 'pi', 'ipi', 'ri', 'serial', 'exp_data'])):
    """Recall protocol.

    Parameters
    ----------
    n_items : int
        List length of learned list.
    pi : float
        Presentation interval in seconds.
    ipi : float
        Interpresentation interval in seconds.
    ri : float
        Retention interval.
    serial : bool
        Indicates serial vs free recall.
    """

    @property
    def pres_phase_duration(self):
        return self.n_items * self.pi + (self.n_items - 1) * self.ipi

    @property
    def duration(self):
        return self.pres_phase_duration + self.ri


class StimulusProvider(object):
    """Recall protocoll.

    Parameters
    ----------
    proto : Recall
        Recall protocoll.
    distractor_rate : float
        Rate of distractors in items per second.
    """
    def __init__(self, proto, distractor_rate):
        self.proto = proto
        self.distractor_rate = distractor_rate

    @staticmethod
    def get_distractor(epoch, i):
        return 'D{epoch}_{i}'.format(epoch=int(epoch), i=int(i))

    @staticmethod
    def get_item(i):
        return 'V' + str(int(i))

    def get_all_items(self):
        return [self.get_item(i) for i in range(self.proto.n_items)]

    def get_all_distractors(self):
        return [self.get_distractor(epoch, i)
                for epoch in range(self.n_epochs)
                for i in range(self.n_distractors_per_epoch)]

    @property
    def n_epochs(self):
        return self.proto.n_items

    @property
    def n_distractors_per_epoch(self):
        return int(np.ceil(
            self.distractor_rate * max(self.proto.ipi, self.proto.ri)))

    def is_pres_phase(self, t):
        return t <= self.proto.pres_phase_duration + self.proto.ri

    def is_recall_phase(self, t):
        return t > self.proto.pres_phase_duration + self.proto.ri

    def make_stimulus_fn(self):
        def stimulus_fn(t):
            if t > self.proto.pres_phase_duration:
                retention_t = t - self.proto.pres_phase_duration
                if retention_t <= self.proto.ri:
                    stimulus = self.get_distractor(
                        epoch=self.proto.n_items - 1,
                        i=int(self.distractor_rate * retention_t))
                else:
                    stimulus = '0'
            else:
                epoch = int(t // (self.proto.pi + self.proto.ipi))
                epoch_t = t % (self.proto.pi + self.proto.ipi)
                if epoch_t <= self.proto.pi:
                    stimulus = self.get_item(epoch)
                else:
                    stimulus = self.get_distractor(
                        epoch=epoch,
                        i=int(self.distractor_rate * (
                            epoch_t - self.proto.pi)))
            return stimulus
        return stimulus_fn

    def stimuli(self, distractor_rate):
        for epoch in range(self.potot.n_items - 1):
            yield self.get_item(epoch)
            for i in range(int(np.ceil(distractor_rate * self.proto.ipi))):
                yield self.get_distractor(epoch=epoch, i=i)
        yield self.get_item(self.proto.n_items - 1)
        for i in range(int(np.ceil(distractor_rate * self.ri))):
            yield self.get_distractor(epoch=self.proto.n_items - 1, i=i)

    def recall_stimuli(self):
        return (self.get_item(i) for i in range(self.proto.n_items))


def _datapath(path):
    return os.path.join(
        os.path.dirname(__file__), '../data/experimental', path)


PROTOCOLS = {
    'serial': Recall(
        n_items=10, pi=1., ipi=0., ri=0., serial=True,
        exp_data=_datapath('Jahnke68/10item_0sec.csv')),
    'immediate': Recall(
        n_items=12, pi=1., ipi=0., ri=0., serial=False,
        exp_data=_datapath('HowaKaha99/Immed.dat')),
    'delayed': Recall(
        n_items=12, pi=1.2, ipi=0., ri=16., serial=False,
        exp_data=_datapath('HowaKaha99/Ltr0.dat')),
    'contdist': Recall(
        n_items=12, pi=1.2, ipi=16., ri=16., serial=False,
        exp_data=_datapath('HowaKaha99/Ltr3.dat')),
}
