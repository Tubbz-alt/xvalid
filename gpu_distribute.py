from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import psutil
import traceback
import StringIO
from mpi4py import MPI

comm = MPI.COMM_WORLD

def gpu_distribute(comm, args):
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    args.gpus = min(world_size, args.gpus)
    
    if rank >= args.gpus:
        # these ranks are done
        comm.Finalize()
        return

    cores_per_gpu=max(1,args.cores//args.gpus)

    p = psutil.Process()
    core0 = rank * cores_per_gpu
    affinity = range(core0, core0+cores_per_gpu)
    p.cpu_affinity(affinity)
    args.gpu = rank
    if args.mock:
        os.environ['MOCK_TENSORFLOW']='1'

    if rank == 0:
        if world_size > args.gpus:
            sys.stderr.write("WARNING: there are more ranks than GPU's, will only use %d ranks\n" % args.gpus)
        if args.cores % args.gpus != 0:
            sys.stderr.write("WARNING: num cores is not a multiple of num gpus, some cores will not be used (this could be fixed)\n")
        if world_size < args.gpus:
            sys.stderr.write("WARNING: world size < num GPU's, later GPU devices will not be used\n")
    sys.stdout.write("rank=%d about to call run()" % rank)
    from pipeline_runner import run
    run(args=args, comm=comm)
    
DESCR='''
distribute works on a many-core host with multiple GPUs.
Takes a pipeline-runner script, a script that will loop through calling a pipeline, and
calls this script once for each GPU, dividing the number of cores evenly. The pipeline script
will limit itself to the prescribed GPU on the host.'''

def main(comm):
    rank = comm.Get_rank()
    if rank == 0:
        NUM_GPUS=8
        NUM_JOBS=1
        CPU_COUNT=psutil.cpu_count()
        assert isinstance(CPU_COUNT,int)
        parser = argparse.ArgumentParser(description=DESCR)
        parser.add_argument('--gpus', type=int, help="number of GPU's on the machine (def=%d)" % NUM_GPUS, default=NUM_GPUS)
        parser.add_argument('--jobs', type=int, help="number of jobs to do (def=%d)" % NUM_JOBS, default=NUM_JOBS)
        parser.add_argument('--seed', type=int, help="global seed for all pipeline jobs", default=23824389)
        parser.add_argument('--cores', type=int, help="cores to use (default all on machine)", default=CPU_COUNT)
        parser.add_argument('--pyscript', type=str, help="pipeline script to run", default='pipeline-runner.py')
        parser.add_argument('--prefix', type=str, help="jobs prefix", default='test')
        parser.add_argument('--mock', action='store_true', help="mock tensorflow for development")
        parser.add_argument('--force', action='store_true', help="overwrite config files, and pipeline step files")
        parser.add_argument('--redoall', action='store_true', help="have jobs redo all steps")
        parser.add_argument('--dev', action='store_true', help="develop mode")
        parser.add_argument('--clean', action='store_true', help="have jobs clean output files")
        parser.add_argument('--log', type=str, help='one of DEBUG,INFO,WARN,ERROR,CRITICAL.', default='INFO')
        
        args = parser.parse_args()
        assert args.pyscript, "must provide a python script to run"
        assert args.prefix, "must provide a prefix to pass to the python script"
        del parser
    else:
        args = None
    comm.Barrier()
    args = comm.bcast(args, root=0)
    gpu_distribute(comm, args)

if __name__ == '__main__':
    try:
        main(comm)
    except Exception:
        rank = comm.Get_rank()
        msg = "rank=%d - exception aborting" % rank
        if rank in [0,1]:
            exceptBuffer = StringIO.StringIO()
            traceback.print_exc(file=exceptBuffer)
            msg += ':\n%s' % exceptBuffer.getvalue()
        sys.stderr.write(msg)
        MPI.COMM_WORLD.Abort(1)
        sys.exit(1)

    
