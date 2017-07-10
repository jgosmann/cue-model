import numpy as np


class Recall(object):
    """Recall protocoll.

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
    distractor_rate : float
        Rate of distractors in items per second.
    serial : bool
        Indicates serial vs free recall.
    """
    def __init__(self, n_items, pi, ipi, ri, distractor_rate, serial=False):
        self.n_items = n_items
        self.pi = pi
        self.ipi = ipi
        self.ri = ri
        self.distractor_rate = distractor_rate
        self.serial = serial
        self.pres_phase_duration = (
            self.n_items * self.pi + (self.n_items - 1) * self.ipi)
        self.duration = self.pres_phase_duration + self.ri

    @staticmethod
    def get_distractor(epoch, i):
        return 'D{epoch}_{i}'.format(epoch=int(epoch), i=int(i))

    @staticmethod
    def get_item(i):
        return 'V' + str(int(i))

    def get_all_items(self):
        return [self.get_item(i) for i in range(self.n_items)]

    def get_all_distractors(self):
        return [self.get_distractor(epoch, i)
                for epoch in range(self.n_epochs)
                for i in range(self.n_distractors_per_epoch)]

    @property
    def n_epochs(self):
        return self.n_items

    @property
    def n_distractors_per_epoch(self):
        return int(np.ceil(self.distractor_rate * max(self.ipi, self.ri)))

    def is_pres_phase(self, t):
        return t <= self.pres_phase_duration + self.ri

    def is_recall_phase(self, t):
        return t > self.pres_phase_duration + self.ri

    def make_stimulus_fn(self):
        def stimulus_fn(t):
            if t > self.pres_phase_duration:
                retention_t = t - self.pres_phase_duration
                if retention_t <= self.ri:
                    stimulus = self.get_distractor(
                        epoch=self.n_items - 1,
                        i=int(self.distractor_rate * retention_t))
                else:
                    stimulus = '0'
            else:
                epoch = int(t // (self.pi + self.ipi))
                epoch_t = t % (self.pi + self.ipi)
                if epoch_t <= self.pi:
                    stimulus = self.get_item(epoch)
                else:
                    stimulus = self.get_distractor(
                        epoch=epoch,
                        i=int(self.distractor_rate * (epoch_t - self.pi)))
            return stimulus
        return stimulus_fn

    def stimuli(self, distractor_rate):
        for epoch in range(self.n_items - 1):
            yield self.get_item(epoch)
            for i in range(int(np.ceil(distractor_rate * self.ipi))):
                yield self.get_distractor(epoch=epoch, i=i)
        yield self.get_item(self.n_items - 1)
        for i in range(int(np.ceil(distractor_rate * self.ri))):
            yield self.get_distractor(epoch=self.n_items - 1, i=i)

    def recall_stimuli(self):
        return (self.get_item(i) for i in range(self.n_items))
