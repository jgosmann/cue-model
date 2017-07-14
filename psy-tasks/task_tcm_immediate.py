import platform

import numpy as np

from psyrun import Param
from psyrun.scheduler import Slurm, Sqsub

from imem.models.imem import IMemTrial


rng = np.random.RandomState(846777)
n_trials = 100
seeds = range(100)


pspace = Param(seed=seeds, trial=range(n_trials))
min_items = 1
max_jobs = 100

sharcnet_nodes = ['narwhal', 'bul', 'kraken', 'saw']
if any(platform.node().startswith(x) for x in sharcnet_nodes):
    workdir = '/work/jgosmann/tcm'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '8h',
        'memory': '4G'
    }
elif platform.node().startswith('gra'):
    workdir = '/project/jgosmann/tcm'
    scheduler = Slurm(workdir)
    scheduler_args = {
        'timelimit': '0-03:00',
        'memory': '2G'
    }


def execute(trial, **kwargs):
    kwargs['protocol'] = 'immed'
    return IMemTrial().run(**kwargs)
