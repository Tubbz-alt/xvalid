from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import time
import psutil
import argparse
import hashlib
if os.environ.get('MOCK_TENSORFLOW',False):
    import psmlearn.mock_tensorflow as tf
else:
    import tensorflow as tf
from mpi4py import MPI

import pipeline_vgg16_xtcav_full as pipelinextcav

comm = MPI.COMM_WORLD

def run(args, comm=comm):
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    with tf.device('/gpu:%d' % args.gpu):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement=False
        config.inter_op_parallelism_threads=args.cores
        config.intra_op_parallelism_threads=args.cores
        # if I don't allow soft placement, I crash - there are certain ops/variables, like the
        # optimizer global step, that don't have a GPU kernel
        config.allow_soft_placement=True
#        config.use_per_session_threads=True

        assert os.path.exists('config_files'), "directory for config files doesn't exist"
        with tf.Session(config=config) as sess:
            md5 = hashlib.md5()
            hostname = os.environ['HOSTNAME']
            pid = psutil.Process().pid
            aff = psutil.Process().cpu_affinity()
            print("rank=%d gpu=%d jobs=%d cores=%d prefix=%s pid=%s affinity=[%d-%d]" %
                  (rank, args.gpu, args.jobs, args.cores, args.prefix, pid, aff[0], aff[-1]))
            for job in range(args.jobs):
                if job % args.gpus != rank: continue
                str_seed = '%s-%d-%d-%s' % (hostname, job, args.gpu, time.time())
                md5.update(str_seed)
                job_seed = int(md5.hexdigest(),16) % 4294967295
                split_seed = int(args.seed) % 4294967295
                configFile = os.path.join('config_files','config_%s_%3.3d.yaml' % (args.prefix, job))
                pipelinextcav.generate_config_file(seed=job_seed, output=configFile, force=args.force)
                argv = [pipelinextcav.__file__,
                        '--split_seed', str(split_seed),
                        '--job_seed', str(job_seed),
                        '--log', args.log]
                if args.force:
                    argv.append('--force')
                if args.dev or args.mock:
                    argv.append('--dev')
                if args.redoall:
                    argv.append('--redoall')
                if args.clean:
                    argv.append('--clean')
                argv.extend(['--config', configFile])
                job_prefix = '%s_job%3.3d' % (args.prefix, job)
                argv.append(job_prefix)
                print("rank=%d job_seed=%s argv=%s" % (rank, job_seed, argv))
                sys.argv = argv
                pipelinextcav.run(argv, comm, sess)
        

DESCR='''Repeatedly runs a pipeline
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help="which gpu to use", default=0)
    parser.add_argument('--jobs', type=int, help="number of jobs to do", default=1)
    parser.add_argument('--cores', type=int, help="number of cores to restrict usage too (default is all)", default=0)
    parser.add_argument('--prefix', type=str, help="jobs prefix", default='test')
    parser.add_argument('--dev', action='store_true', help="development mode")
    args = parser.parse_args()
    run(args)
    
