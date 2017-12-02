import platform

import numpy as np

from psyrun import Param
from psyrun.scheduler import Slurm, Sqsub

from imem.model.trial import IMemTrial


rng = np.random.RandomState(846777)
n_trials = 100
seeds = range(100)


pspace = Param(seed=seeds, trial=range(n_trials))
min_items = 1
pool_size = 1
max_jobs = 100

if platform.node().startswith('gra') or platform.node().startswith('cedar'):
    workdir = '/scratch/jgosmann/tcm'
    scheduler = Slurm(workdir)
    def timelimit(name):
        if 'split' in name or 'merge' in name:
            return '0-00:10'
        else:
            return '0-12:00'
    def memory_per_cpu(name):
        if 'split' in name:
            return '300M'
        elif 'merge' in name:
            return '60M'
        else:
            return '4G'
    scheduler_args = {
        'timelimit': timelimit,
        'n_cpus': '1',
        'memory_per_cpu': memory_per_cpu,
    }

def execute(trial, **kwargs):
    kwargs['protocol'] = 'contdist'
    result = IMemTrial().run(**kwargs)
    return result
