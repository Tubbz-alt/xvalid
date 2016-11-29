from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

import relevance_propagation_vgg16 as rp

def assert_almost_equal(A,B, msg='', tol=1e-1, verbose=True):
    assert A.shape==B.shape, "A.shape=%s B.shape=%s" % (A.shape, B.shape)
    diff = np.sum(np.abs(A-B))
    if verbose:
        assert diff < tol, "FAIL %s: diff=%.4f\nA=%s\nB=%s" % (msg, diff,A,B)
    else:
        assert diff < tol, "FAIL %s: diff=%.4f" % (msg, diff)

class TestRelProp(object):
    def __init__(self, sess):
        self.sess=sess
        self.naive = rp.Naive()
        self.epsVariant = rp.EpsVariant(eps=1e-3)
        self.betaVariant = rp.BetaVariant(beta=-.1)
    
class TestScalarFn(TestRelProp):
    def __init__(self, sess, N=4):
        TestRelProp.__init__(self, sess)
        self.N=N        
        self.W=np.ones((self.N,4), dtype=np.float32)
        self.W *= np.random.normal(size=(self.N,4))
        self.X = tf.placeholder(tf.float32, (None, self.N))
        self.Xarr = np.maximum(0,np.random.normal(size=(10,self.N)).astype(np.float32))
        self.fn = 10.0;
        self.test_naive(self.Xarr, np.maximum(.1, self.W), self.naive)
        self.test_eps(self.Xarr, self.W, self.epsVariant)
        self.test_beta(self.Xarr, self.W, self.betaVariant)
    
    def test_naive(self, Xarr, Warr, rule):
        tfR, tfZ_norm, tfZ = rp.relprop_from_scalar_fn(self.X[0], Warr[:,3],
                                                       self.fn, rule,
                                                       debug=True)
        tfR_arr, tfZ_norm_arr, tfZ_arr =self.sess.run([tfR, tfZ_norm, tfZ],
                                                      feed_dict={self.X:Xarr})

        Z = Xarr[0,:] * Warr[:,3]
        Z_sum = np.sum(Z)
        Rnaive = Z / Z_sum
        Rnaive *= self.fn

        assert_almost_equal(Z, tfZ_arr)
        assert_almost_equal(Z_sum, tfZ_norm_arr)
        assert_almost_equal(Rnaive, tfR_arr)
        print("OK")
        
    def test_eps(self, Xarr, Warr, rule):
        tfR, tfZ_norm, tfZ = rp.relprop_from_scalar_fn(self.X[0], Warr[:,3], self.fn,
                                                       rule, debug=True)

        tfR_arr, tfZ_norm_arr, tfZ_arr = self.sess.run([tfR, tfZ_norm, tfZ],
                                                       feed_dict={self.X:Xarr})
        Z = Xarr[0,:] * Warr[:,3]
        Z_sum = np.sum(Z)
        Z_norm = (Z_sum + rule.eps * np.sign(Z_sum))
        Reps = Z / Z_norm
        Reps *= self.fn
    
        assert_almost_equal(Z, tfZ_arr)
        assert_almost_equal(Z_norm, tfZ_norm_arr, "eps: Z_norm != tfZ_norm_arr")
        assert_almost_equal(Reps, tfR_arr)

    def test_beta(self, Xarr, Warr, rule):
        tfR, tfZneg_sum, tfZpos_sum, tfZneg, tfZpos, tfZ = \
                rp.relprop_from_scalar_fn(self.X[0], Warr[:,3], self.fn, rule, debug=True)

        tfR_arr, tfZneg_sum_arr, tfZpos_sum_arr, tfZneg_arr, tfZpos_arr, tfZ_arr = \
                self.sess.run([tfR, tfZneg_sum, tfZpos_sum, tfZneg, tfZpos, tfZ],
                              feed_dict={self.X:Xarr})

        Z = Xarr[0,:] * Warr[:,3]
        Zpos = np.maximum(0,Z)
        Zneg = np.minimum(0,Z)
        Zpos_sum = np.sum(Zpos)
        Zneg_sum = np.sum(Zneg)

        Rbeta = np.zeros_like(Zpos)
        if Zpos_sum > 1e-8:
            Rbeta += Zpos * (rule.alpha/Zpos_sum)
        if Zneg_sum < -1e-8:
            Rbeta += Zneg * (rule.beta/Zneg_sum)
        Rbeta *= self.fn
    
        assert_almost_equal(Z, tfZ_arr)
        assert_almost_equal(Rbeta, tfR_arr)

class TestDenseLayer(TestRelProp):
    def __init__(self, sess, N=4, M=5):
        TestRelProp.__init__(self, sess)
        self.N=N
        self.M=M
        
        W=tf.ones((N,M), dtype=tf.float32)
        W=tf.mul(W, np.random.normal(size=(N,M)).astype(np.float32))

        RprevNaive = tf.ones((M,))
        Rprev = tf.mul(tf.ones(M,), np.random.normal(size=(M,)).astype(np.float32))

        self.X = tf.placeholder(tf.float32, (None, self.N))
        Xarr = np.maximum(0.0,np.random.normal(size=(10,self.N)).astype(np.float32))

        self.test_naive(Xarr, 0.1 + tf.nn.relu(W), RprevNaive, self.naive)
        self.test_eps(Xarr, W, Rprev, self.epsVariant)
        self.test_beta(Xarr, W, Rprev, self.betaVariant)

    def test_naive(self, Xarr, W, Rprev, rule):
        tfR, tfZ_norm, tfZ = rp.relprop_from_fully_connected(self.X[0], W,
                                                             Rprev, rule, debug=True)
        tfR_arr, tfZ_norm_arr, tfZ_arr = self.sess.run([tfR, tfZ_norm, tfZ], feed_dict={self.X:Xarr})
        W_arr = W.eval()
        Rprev_arr = Rprev.eval()

        Z_arr = np.reshape(Xarr[0,:],(self.N,1)) * W_arr
        Z_reduce_arr = np.sum(Z_arr, axis=0)
        Z_reduce_arr = np.reshape(Z_reduce_arr, (1,self.M))
        Z_norm_arr = Z_arr / Z_reduce_arr
        R_arr = np.matmul(Z_norm_arr, Rprev_arr)

        assert_almost_equal(Z_arr, tfZ_arr)
        assert_almost_equal(R_arr, tfR_arr)

    def test_eps(self, Xarr, W, Rprev, rule):
        tfR, tfZ_norm, tfZ = rp.relprop_from_fully_connected(self.X[0], W, Rprev,
                                                             rule, debug=True)
        tfR_arr, tfZ_norm_arr, tfZ_arr = self.sess.run([tfR, tfZ_norm, tfZ], feed_dict={self.X:Xarr})

        W_arr = W.eval()
        Rprev_arr = Rprev.eval()

        Z_arr = np.reshape(Xarr[0,:],(self.N,1)) * W_arr
        Z_reduce_arr = np.sum(Z_arr, axis=0)
        Z_reduce_arr += rule.eps * np.sign(Z_reduce_arr)
        Z_reduce_arr = np.reshape(Z_reduce_arr, (1,self.M))
        Z_norm_arr = Z_arr / Z_reduce_arr
        R_arr = np.matmul(Z_norm_arr, Rprev_arr)
        assert_almost_equal(Z_arr, tfZ_arr)
        assert_almost_equal(R_arr, tfR_arr)

    def test_beta(self, Xarr, W, Rprev, rule):
        tfR = rp.relprop_from_fully_connected(self.X[0], W, Rprev, rule)
        tfR_arr = self.sess.run(tfR, feed_dict={self.X:Xarr})
        Warr = W.eval()
        Rprev_arr = Rprev.eval()
        Z_arr = np.reshape(Xarr[0,:],(self.N,1)) * Warr
        Z_pos = np.maximum(Z_arr, 0)
        Z_neg = np.minimum(Z_arr, 0)
        Z_pos_reduce = np.reshape(np.maximum(1e-10, np.sum(Z_pos,0)), (1,self.M))
        Z_neg_reduce = np.reshape(np.minimum(-1e-10, np.sum(Z_neg,0)), (1,self.M))

        Z_norm = Z_pos * rule.alpha / Z_pos_reduce
        Z_norm += Z_neg * rule.beta / Z_neg_reduce

        R_arr = np.matmul(Z_norm, Rprev_arr)
        assert_almost_equal(R_arr, tfR_arr)

class TestConv(TestRelProp):
    def __init__(self, sess, N=4, M=5, Ch_a=2, Ch_b=3):
        TestRelProp.__init__(self, sess)
        K=tf.ones((2,2,Ch_a,Ch_b))
        K *= np.random.normal(size=(2,2,Ch_a,Ch_b))
        strides=(1,1,1,1)
        X = tf.placeholder(tf.float32, (None, N, M, Ch_a))
        
        Y = tf.nn.conv2d(X, K, strides, padding='SAME')
        Rprev = tf.ones_like(Y[0])
        Rprev *= np.random.normal(size=(map(int,Rprev.get_shape())))

        R = rp.relprop_from_conv(X,Y,Rprev, self.epsVariant, K, strides)

        Xarr = np.random.normal(size=(1, N, M, Ch_a))
        Rarr = self.sess.run(R, feed_dict={X:Xarr})
        print(R)
        print(Rarr)
        
def test_relprop_max(sess, B=3,N=6,M=8,C=2):
    X = tf.placeholder(tf.float32, (None,N,M,C))
    Y = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    Rprev = tf.ones((1,3,4,2), tf.float32)
    Rprev *= np.random.normal(size=(1,3,4,2))

    R = rp.relprop_from_max(X,Y,Rprev)
    Xarr = np.maximum(0.0, np.random.normal(size=(1,N,M,C)))
    Xarr[0,0,0,0]=.3
    Xarr[0,0,1,0]=.5
    Xarr[0,1,0,0]=.6
    Xarr[0,1,1,0]=.4
    Rarr = sess.run(R,feed_dict={X:Xarr})
    assert_almost_equal(np.sum(Rarr), np.sum(Rprev.eval()))
    assert Rarr[0,0,0,0]==0
    assert Rarr[0,0,1,0]==0
    assert Rarr[0,1,0,0]==Rprev.eval()[0,0,0,0], "Rarr[0,1,0,0]=%.3f != Rprev[0,0,0,0]=%.3f" % (Rarr[0,1,0,0], Rprev.eval()[0,0,0,0])
    assert Rarr[0,1,1,0]==0

def test_relprop(sess):
#    TestScalarFn(sess)
#    TestDenseLayer(sess)
    test_relprop_max(sess)
    TestConv(sess)
    
if __name__ == '__main__':
    pass
