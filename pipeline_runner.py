from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import argparse
if os.environ['DEV']:
    import mock_tensorflow as tf
else:
    import tensorflow as tf

def run_sess(args, config, sess):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    op = tf.matmul(a, b)
    hostname = os.environ['HOSTNAME']
    for job in range(args.jobs):
        if job % args.jobs != args.gpu: continue
        seed = '%s-%d-%d-%s' % (hostname, job, args.gpu, time.time())
        print(seed)
        
def run(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=True
    config.inter_op_parallelism_threads=args.cores,
    config.intra_op_parallelism_threads=args.cores,
#    config.use_per_session_threads=True

    with tf.Session(config=config) as sess:
        run_sess(args, config, sess)

DESCR='''Repeatedly launches a pipeline
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, help="number of jobs to do", default=1)
    parser.add_argument('--gpu', type=int, help="which gpu to use", default=0)
    parser.add_argument('--cores', type=int, help="number of cores to restrict usage too (default is all)", default=0)
    parser.add_argument('--prefix', type=str, help="jobs prefix", default='test')
    parser.add_argument('--dev', action='store_true', help="development mode")
    args = parser.parse_args()
    with tf.device('/gpu:%d' % args.gpu):
        run(args)

    
