""" Code for loading data and generating data batches during training """
from __future__ import division

import copy
import logging
import os
import glob
import tempfile
import pickle
from datetime import datetime
from collections import OrderedDict

import numpy as np
import random
import tensorflow as tf
from utils import extract_demo_dict, Timer
from tensorflow.python.platform import flags
from natsort import natsorted
from random import shuffle

FLAGS = flags.FLAGS

class DataGenerator(object):
    def __init__(self, config={}):
        # Hyperparameters
        self.update_batch_size = FLAGS.update_batch_size
        self.test_batch_size = FLAGS.train_update_batch_size if FLAGS.train_update_batch_size != -1 else self.update_batch_size
        self.meta_batch_size = FLAGS.meta_batch_size
        self.T = FLAGS.T
        self.demo_gif_dir = FLAGS.demo_gif_dir
        self.gif_prefix = FLAGS.gif_prefix
        self.restore_iter = FLAGS.restore_iter
        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        demo_file = FLAGS.demo_file
        demo_file = natsorted(glob.glob(demo_file + '/*pkl'))
        self.dataset_size = len(demo_file)
        if FLAGS.train and FLAGS.training_set_size != -1:
            tmp = demo_file[:FLAGS.training_set_size]
            tmp.extend(demo_file[-FLAGS.val_set_size:])
            demo_file = tmp
        self.extract_supervised_data(demo_file)
        if FLAGS.use_noisy_demos:
            self.noisy_demo_gif_dir = FLAGS.noisy_demo_gif_dir
            noisy_demo_file = FLAGS.noisy_demo_file
            self.extract_supervised_data(noisy_demo_file, noisy=True)

    def extract_supervised_data(self, demo_file, noisy=False):
        """
            Load the states and actions of the demos into memory.
            Args:
                demo_file: list of demo files where each file contains expert's states and actions of one task.
        """
        demos = extract_demo_dict(demo_file)
        # We don't need the whole dataset of simulated pushing.
        if FLAGS.experiment == 'sim_push':
            for key in demos.keys():
                demos[key]['demoX'] = demos[key]['demoX'][6:-6, :, :].copy()
                demos[key]['demoU'] = demos[key]['demoU'][6:-6, :, :].copy()
        n_folders = len(demos.keys())
        N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.iteritems())
        self.state_idx = range(demos[0]['demoX'].shape[-1])
        self._dU = demos[0]['demoU'].shape[-1]
        print "Number of demos: %d" % N_demos
        idx = np.arange(n_folders)
        if FLAGS.train:
            n_val = FLAGS.val_set_size # number of demos for testing
            if not hasattr(self, 'train_idx'):
                if n_val != 0:
                    if not FLAGS.shuffle_val:
                        self.val_idx = idx[-n_val:]
                        self.train_idx = idx[:-n_val]
                    else:
                        self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
                        mask = np.array([(i in self.val_idx) for i in idx])
                        self.train_idx = np.sort(idx[~mask])
                else:
                    self.train_idx = idx
                    self.val_idx = []
            # Normalize the states if it's training.
            with Timer('Normalizing states'):
                if self.scale is None or self.bias is None:
                    states = np.vstack((demos[i]['demoX'] for i in self.train_idx)) # hardcoded here to solve the memory issue
                    states = states.reshape(-1, len(self.state_idx))
                    # 1e-3 to avoid infs if some state dimensions don't change in the
                    # first batch of samples
                    self.scale = np.diag(
                        1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                    self.bias = - np.mean(
                        states.dot(self.scale), axis=0)
                    # Save the scale and bias.
                    with open('data/scale_and_bias_%s.pkl' % FLAGS.experiment, 'wb') as f:
                        pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
                for key in demos.keys():
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.state_idx))
                    demos[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.state_idx))
        if not noisy:
            self.demos = demos
        else:
            self.noisy_demos = demos

    def generate_batches(self, noisy=False):
        with Timer('Generating batches for each iteration'):
            if FLAGS.training_set_size != -1:
                offset = self.dataset_size - FLAGS.training_set_size - FLAGS.val_set_size
            else:
                offset = 0
            train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.train_idx}
            val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % (i+offset)) for i in self.val_idx}
            if noisy:
                noisy_train_img_folders = {i: os.path.join(self.noisy_demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.train_idx}
                noisy_val_img_folders = {i: os.path.join(self.noisy_demo_gif_dir, self.gif_prefix + '_%d' % (i+offset)) for i in self.val_idx}
            TEST_PRINT_INTERVAL = 500
            TOTAL_ITERS = FLAGS.metatrain_iterations
            self.all_training_filenames = []
            self.all_val_filenames = []
            self.training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
            self.val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, int(TOTAL_ITERS/TEST_PRINT_INTERVAL))}
            if noisy:
                self.noisy_training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
                self.noisy_val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, TOTAL_ITERS/TEST_PRINT_INTERVAL)}
            for itr in xrange(TOTAL_ITERS):
                sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
                for idx in sampled_train_idx:
                    sampled_folder = train_img_folders[idx]
                    image_paths = natsorted(os.listdir(sampled_folder))
                    if FLAGS.experiment == 'sim_push':
                        image_paths = image_paths[6:-6]
                    try:
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    if noisy:
                        noisy_sampled_folder = noisy_train_img_folders[idx]
                        noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                        assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                    if not noisy:
                        sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                        sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                    else:
                        noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) #True
                        sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) #True
                        sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                        sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                    self.all_training_filenames.extend(sampled_images)
                    self.training_batch_idx[itr][idx] = sampled_image_idx
                    if noisy:
                        self.noisy_training_batch_idx[itr][idx] = noisy_sampled_image_idx
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                    for idx in sampled_val_idx:
                        sampled_folder = val_img_folders[idx]
                        image_paths = natsorted(os.listdir(sampled_folder))
                        if FLAGS.experiment == 'sim_push':
                            image_paths = image_paths[6:-6]
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                        if noisy:
                            noisy_sampled_folder = noisy_val_img_folders[idx]
                            noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                            assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                        if not noisy:
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                            sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                        else:
                            noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) # True
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) # True
                            sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                            sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                        self.all_val_filenames.extend(sampled_images)
                        self.val_batch_idx[itr][idx] = sampled_image_idx
                        if noisy:
                            self.noisy_val_batch_idx[itr][idx] = noisy_sampled_image_idx

    def make_batch_tensor(self, network_config, restore_iter=0, train=True):
        TEST_INTERVAL = 500
        batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size
        if train:
            all_filenames = self.all_training_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
        else:
            all_filenames = self.all_val_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(int(restore_iter/TEST_INTERVAL)+1):]
        
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print 'Generating image processing ops'
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.T, im_height, im_width, num_channels))
        image = tf.cast(image, tf.float32)
        image /= 255.0
        if FLAGS.hsv:
            eps_min, eps_max = 0.5, 1.5
            assert eps_max >= eps_min >= 0
            # convert to HSV only fine if input images in [0, 1]
            img_hsv = tf.image.rgb_to_hsv(image)
            img_h = img_hsv[..., 0]
            img_s = img_hsv[..., 1]
            img_v = img_hsv[..., 2]
            eps = tf.random_uniform([self.T, 1, 1], eps_min, eps_max)
            img_v = tf.clip_by_value(eps * img_v, 0., 1.)
            img_hsv = tf.stack([img_h, img_s, img_v], 3)
            image_rgb = tf.image.hsv_to_rgb(img_hsv)
            image = image_rgb
        image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 64 #128 #256
        print 'Batching images'
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []
        for i in xrange(self.meta_batch_size):
            image = images[i*(self.update_batch_size+self.test_batch_size):(i+1)*(self.update_batch_size+self.test_batch_size)]
            image = tf.reshape(image, [(self.update_batch_size+self.test_batch_size)*self.T, -1])
            all_images.append(image)
        return tf.stack(all_images)
        
    def generate_data_batch(self, itr, train=True):
        if train:
            demos = {key: self.demos[key].copy() for key in self.train_idx}
            idxes = self.training_batch_idx[itr]
            if FLAGS.use_noisy_demos:
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.train_idx}
                noisy_idxes = self.noisy_training_batch_idx[itr]
        else:
            demos = {key: self.demos[key].copy() for key in self.val_idx}
            idxes = self.val_batch_idx[itr]
            if FLAGS.use_noisy_demos:
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.val_idx}
                noisy_idxes = self.noisy_val_batch_idx[itr]
        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        test_batch_size = self.test_batch_size
        if not FLAGS.use_noisy_demos:
            U = [demos[k]['demoU'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
            U = np.array(U)
            X = [demos[k]['demoX'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
            X = np.array(X)
        else:
            noisy_U = [noisy_demos[k]['demoU'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            noisy_X = [noisy_demos[k]['demoX'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            U = [demos[k]['demoU'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            U = np.concatenate((np.array(noisy_U), np.array(U)), axis=1)
            X = [demos[k]['demoX'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            X = np.concatenate((np.array(noisy_X), np.array(X)), axis=1)
        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.state_idx)
        return X, U
