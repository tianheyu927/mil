""" Utility functions for tensorflow. """

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
import numpy as np

def safe_get(name, *args, **kwargs):
    """ Same as tf.get_variable, except flips on reuse_variables automatically """
    try:
        return tf.get_variable(name, *args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(name, *args, **kwargs)

def init_weights(shape, name=None):
    shape = tuple(shape)
    weights = np.random.normal(scale=0.01, size=shape).astype('f')
    return safe_get(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)
    
def init_bias(shape, name=None):
    return safe_get(name, initializer=tf.zeros(shape, dtype=tf.float32))

def init_fc_weights_xavier(shape, name=None):
    fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    return safe_get(name, list(shape), initializer=fc_initializer, dtype=tf.float32)

def init_conv_weights_xavier(shape, name=None):
    conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    return safe_get(name, list(shape), initializer=conv_initializer, dtype=tf.float32)
    
def init_fc_weights_snn(shape, name=None):
    weights = np.random.normal(scale=np.sqrt(1.0/shape[0]), size=shape).astype('f')
    return safe_get(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)

def init_conv_weights_snn(shape, name=None):
    weights = np.random.normal(scale=np.sqrt(1.0/(shape[0]*shape[1]*shape[2])), size=shape).astype('f')
    return safe_get(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)

def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result

def euclidean_loss_layer(a, b, multiplier=100.0, use_l1=False, eps=0.01):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    multiplier = tf.constant(multiplier, dtype='float') #for bc #10000
    uP =a*multiplier-b*multiplier
    if use_l1:
        return tf.reduce_mean(eps*tf.square(uP) + tf.abs(uP))
    return tf.reduce_mean(tf.square(uP))

def conv2d(img, w, b, strides=[1, 1, 1, 1], is_dilated=False):
    if is_dilated:
        layer = tf.nn.atrous_conv2d(img, w, rate=2, padding='SAME') + b
    else:
        layer = tf.nn.conv2d(img, w, strides=strides, padding='SAME') + b
    return layer
    
def conv1d(img, w, b, stride=1):
    layer = tf.nn.conv1d(img, w, stride=stride, padding='SAME') + b
    return layer
            
def dropout(layer, keep_prob=0.9, is_training=True, name=None, selu=False):
    if selu:
        return dropout_selu(layer, 1.0 - keep_prob, name=name, training=is_training)
    if is_training:
        return tf.nn.dropout(layer, keep_prob=keep_prob, name=name)
    else:
        return tf.add(layer, 0, name=name)

def norm(layer, norm_type='batch_norm', decay=0.9, id=0, is_training=True, activation_fn=tf.nn.relu, prefix='conv_'):
    if norm_type != 'batch_norm' and norm_type != 'layer_norm':
        return tf.nn.relu(layer)
    with tf.variable_scope('norm_layer_%s%d' % (prefix, id)) as vs:
        if norm_type == 'batch_norm':
            if is_training:
                try:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                        scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs) # updates_collections=None
                except ValueError:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                        scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True) # updates_collections=None
            else:
                layer = tf.contrib.layers.batch_norm(layer, is_training=False, center=True,
                    scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True) # updates_collections=None
        elif norm_type == 'layer_norm': # layer_norm
            # Take activation_fn out to apply lrelu
            try:
                layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                    scale=False, scope=vs)) # updates_collections=None
                
            except ValueError:
                layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                    scale=False, scope=vs, reuse=True))
        elif norm_type == 'selu':
            layer = selu(layer)
        else:
            raise NotImplementedError('Other types of norm not implemented.')
        return layer
        
class VBN(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.name = name
            self.mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
            self.batch_size = int(x.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = tf.nn.relu(self._normalize(x, self.mean, self.mean_sq, "reference"))
            self.reference_output = out

    def __call__(self, x, update=False):
        with tf.variable_scope(self.name) as scope:
            if not update:
                new_coeff = 1. / (self.batch_size + 1.)
                old_coeff = 1. - new_coeff
                new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
                new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keep_dims=True)
                mean = new_coeff * new_mean + old_coeff * self.mean
                mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
                out = tf.nn.relu(self._normalize(x, mean, mean_sq, "live"))
            # Update the mean and mean_sq when passing the reference data
            else:
                self.mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
                self.mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
                out = tf.nn.relu(self._normalize(x, self.mean, self.mean_sq, "reference"))
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma = safe_get("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
        self.beta = safe_get("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        out = out * gamma
        out = out + beta
        return out

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Consider stride size when using xavier for fp network
def get_xavier_weights(filter_shape, poolsize=(2, 2), name=None):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    weights = np.random.uniform(low=low, high=high, size=filter_shape)
    return safe_get(name, filter_shape, initializer=tf.constant_initializer(weights))

def get_he_weights(filter_shape, name=None):
    fan_in = np.prod(filter_shape[1:])

    stddev = np.sqrt(2.6/fan_in)
    weights = stddev * np.random.randn(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3])
    return safe_get(name, filter_shape, initializer=tf.constant_initializer(weights))
