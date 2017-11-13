""" General utiliy functions """
import logging
try:
   import cPickle as pickle
except:
   import pickle
import gzip
import contextlib
import numpy as np
import scipy.ndimage as sp_ndimage
import os
import errno
import time
import traceback as tb

from color_print import *


LOGGER = logging.getLogger(__name__)

@contextlib.contextmanager
def open_zip(filename, mode='r'):
    """
    Open a file; if filename ends with .gz, opens as a gzip file
    """
    if filename.endswith('.gz'):
        openfn = gzip.open
    else:
        openfn = open
    yield openfn(filename, mode)

class DataLogger(object):
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """
    def __init__(self):
        pass

    def pickle(self, filename, data):
        """ Pickle data into file specified by filename. """
        with open_zip(filename, 'wb') as f:
            pickle.dump(data, f)

    def unpickle(self, filename):
        """ Unpickle data from file specified by filename. """
        try:
            with open_zip(filename, 'rb') as f:
                result = pickle.load(f)
            return result
        except IOError:
            LOGGER.debug('Unpickle error. Cannot find file: %s', filename)
            return None
            
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def extract_demo_dict(demo_file):
    if type(demo_file) is not list:
        demos = DataLogger().unpickle(demo_file)
    else:
        demos = {}
        for i in xrange(0, len(demo_file)):
            with Timer('Extracting demo file %d' % i):
                demos[i] = DataLogger().unpickle(demo_file[i])
    return demos

class Timer(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        new_time = time.time() - self.time_start
        fname, lineno, method, _ = tb.extract_stack()[-2]  # Get caller
        _, fname = os.path.split(fname)
        id_str = '%s:%s' % (fname, method)
        print 'TIMER:'+color_string('%s: %s (Elapsed: %fs)' % (id_str, self.message, new_time), color='gray')

def load_scale_and_bias(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        scale = data['scale']
        bias = data['bias']
    return scale, bias
    
def generate_noise(T, dU):
    """
    Generate a T x dU gaussian-distributed noise vector. This will
    approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    Hyperparams:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """
    var = 2.0
    noise = np.random.randn(T, dU)
    # Smooth noise. This violates the controller assumption, but
    # might produce smoother motions.
    for i in range(dU):
        noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
    variance = np.var(noise, axis=0)
    noise = noise / np.sqrt(variance)
    return noise
