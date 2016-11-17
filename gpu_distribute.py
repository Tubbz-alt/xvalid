from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

def xvalidate_gpus(args):
    if args.cores < args.gpus:
        sys.stderr.write("WARNING: there are more GPU's than cores, cores will be oversubscribed\n")
    if args.cores % args.gpus != 0:
        sys.stderr.write("WARNING: num cores is not a multiple of num gpus, some cores will not be used (this could be fixed)\n")
    cores_per_gpu=max(1,args.cores//args.gpus)
    core_affinity_mask = (1<<cores_per_gpu)-1

    for gpu in range(min(args.jobs, args.gpus)):
            taskset_mask = core_affinity_mask << gpu
            cmd = 'taskset 0x%x' % taskset_mask
            cmd += ' python'
            cmd += ' %s' % args.pyscript
            cmd += ' --prefix %s' % args.prefix
            cmd += ' --jobs %d --gpu %d --cores %d' % (args.jobs, gpu, cores_per_gpu)
            if args.dev:
                cmd = 'DEV=1 %s' % cmd
                cmd += ' --dev'
            print(cmd)
            os.system("%s &" % cmd)

DESCR='''
distribute works on a many-core host with multiple GPUs.
Takes a pipeline-runner script, a script that will loop through calling a pipeline, and
calls this script once for each GPU, dividing the number of cores evenly. The pipeline script
will limit itself to the prescribed GPU on the host.'''

if __name__ == '__main__':

    DEFAULT_NUM_GPUS=8
    DEFAULT_NUM_CORES=48
    DEFAULT_NUM_JOBS=1

    parser = argparse.ArgumentParser(description=DESCR)
    parser.add_argument('--gpus', type=int, help="number of GPU's on the machine (def=%d)" % DEFAULT_NUM_GPUS, default=DEFAULT_NUM_GPUS)
    parser.add_argument('--cores', type=int, help="number of core's on the machine (def=%d)" % DEFAULT_NUM_CORES, default=DEFAULT_NUM_CORES)
    parser.add_argument('--jobs', type=int, help="number of jobs to do (def=%d)" % DEFAULT_NUM_JOBS, default=DEFAULT_NUM_JOBS)
    parser.add_argument('--pyscript', type=str, help="pipeline script to run", default='pipeline_runner.py')
    parser.add_argument('--prefix', type=str, help="jobs prefix", default='test')
    parser.add_argument('--dev', action='store_true', help="development mode")
    
    args = parser.parse_args()
    assert args.pyscript, "must provide a python script to run"
    assert args.prefix, "must provide a prefix to pass to the python script"
    xvalidate_gpus(args)
