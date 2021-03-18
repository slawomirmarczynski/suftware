"""
Suftware

An quick and dirty hack: all dependent files are copy-pasted
into this single py-file.
"""

import scipy as sp
import scipy.stats as stats
from scipy import interpolate
import numpy as np
from numpy.polynomial.legendre import legval, legval2d
import pandas as pd
import sys
import time
import pdb
import numbers
import math
import sys
import numbers
import os
from functools import wraps

from scipy.sparse import csr_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import det, eigh, solve, eigvalsh, inv
import scipy.optimize as opt
import time

from __future__ import division
from scipy.sparse import diags
import multiprocessing as mp
import itertools
import time
import sys
from scipy.linalg import solve, det, norm

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import det, eigh, qr

# AT: fixing SciPy comb import bug: 8_19_2019
# for SciPy versions >= 0.19
try:  
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import pickle


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# BIG IMPORTS HERE
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class ConsumedTimeTimer:
    """
    An adapter/facade/strategy to timer functions for various Pythons.

    Since Python 3.8 the old gold time.clock() method does not work. The method
    clock() from module time had been marked as deprecated since Python 3.3.
    After the update/upgrade to Python 3.8 a call of time.clock() raises
    an AttributeError exception the message "module 'time' has no attribute
    'clock'". This make some older code forward incompatible.

    The simple solution would be replacing time.clock() deprecated method with
    time.process_time(). Nethertheless, this may lead to a backward
    incompatible code. Therefore, in order to overcome these difficulties,
    the idea was born to create ConsumedTimeTimer class as an object-oriented
    facade that encapsulate the choice of strategy for selecting
    the appropriate time-related function in the run time.

    @author: Sławomir Marczyński
    """

    def __init__(self, exclude_sleep_time=True):
        """
        Initialize an ConsumedTime object.

        Args:
            exclude_sleep: allows to select whether to prefer measure only
                           the actively spent time (time used by the process),
                           or also the inactivity time (i.e. wall time).
                           Defaults to True.
        Returns:
            None.

        """
        self._last_tic_time = 0  # to avoid an undefined behaviour

        if sys.version_info.major <= 3 and sys.version_info.minor <= 3:
            self.get_time = time.clock  # pylint: disable=no-member
        elif not exclude_sleep_time:
            self.get_time = time.perf_counter
        else:
            self.get_time = time.process_time

    def __call__(self):
        """
        An redefinition of the call operator. This makes using
        ConsumedTimeTimer objects very easy, see example below::

            clock = ConsumedTimeTimer()
            t1 = clock()
            print('currently timer shows', t1, 'seconds')
            t2 = clock()
            print(t2 - t1, 'seconds missed from previous message')

        Notice, that t1 would be probably not equal 0. This is because objects
        ConsumedTimeTimer neither reset clock nor remeber a time offset.

        Returns:
            float: the time as a float point number of seconds.

        """
        return self.get_time()

    def tic(self):
        """
        An method like Matlab toc - use it to start measuring computation
        time::

            clock = ConsumedTimeTimer()
            clock.tic()
            ...
            consumed_time = clock.toc()

        Returns:
            float: time, in seconds, between the call of tic and the call toc
                   methods.
        """
        self._tic = self()
        return self._tic

    def toc(self):
        delta = self() - self._last_tic_time
        # Remove comment to print messages like Matlab do.
        # print('Elapsed time is', delta, 'seconds.')
        return delta

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# A very small floating point number, used to prevent taking logs of 0
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
TINY_FLOAT32 = sp.finfo(sp.float32).tiny
PHI_MIN = -500 
PHI_MAX = 500
PHI_STD_REG = 100.0
LISTLIKE = (list, np.ndarray, np.matrix, range)

# This is useful for testing whether something is a number
#NUMBER = (int, float, long)
NUMBER = (int, float, int)

# This is useful for testing whether something is an array
ARRAY = (np.ndarray, list)

# Dummy class
class Dummy():
    def __init__(self):
        pass

# Define error handling
class ControlledError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


# Evaluate geodesic distance 
def geo_dist(P,Q):

    # Make sure P is valid
    if not all(np.isreal(P)):
        raise ControlledError('/geo_dist/ P is not real: P = %s' % P)
    if not all(np.isfinite(P)):
        raise ControlledError('/geo_dist/ P is not finite: P = %s' % P)
    if not all(P >= 0):
        raise ControlledError('/geo_dist/ P is not non-negative: P = %s' % P)
    if not any(P > 0):
        raise ControlledError('/geo_dist/ P is vanishing: P = %s' % P)
    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/geo_dist/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/geo_dist/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/geo_dist/ Q is not non-negative: Q = %s' % Q)
    if not any(Q > 0):
        raise ControlledError('/geo_dist/ Q is vanishing: Q = %s' % Q)

    # Enforce proper normalization
    P_prob = P/sp.sum(P) 
    Q_prob = Q/sp.sum(Q) 

    # Return geo-distance. Arc-cosine can behave badly if argument is too close to one, so prepare for this
    try:
        dist = 2*sp.arccos(sp.sum(sp.sqrt(P_prob*Q_prob)))
        if not np.isreal(dist):
            raise ControlledError('/geo_dist/ dist is not real: dist = %s' % dist)
        if not (dist >= 0):
            raise ControlledError('/geo_dist/ dist is not >= 0: dist = %s' % dist)
    except:
        if sp.sum(sp.sqrt(P_prob*Q_prob)) > 1 - TINY_FLOAT32:
            dist = 0
        else:
            raise ControlledError('/geo_dist/ dist cannot be computed correctly!')

    # Return geo-distance
    return dist


# Convert field to non-normalized quasi probability distribution
def field_to_quasiprob(raw_phi):
    phi = np.copy(raw_phi) 
    G = len(phi)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/field_to_quasiprob/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/field_to_quasiprob/ phi is not finite: phi = %s' % phi)
    
    if any(phi < PHI_MIN):
        phi[phi < PHI_MIN] = PHI_MIN      

    # Compute quasiQ
    quasiQ = sp.exp(-phi)/(1.*G)

    # Make sure quasiQ is valid
    if not all(np.isreal(quasiQ)):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not real: quasiQ = %s' % quasiQ)
    if not all(np.isfinite(quasiQ)):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not finite: quasiQ = %s' % quasiQ)
    if not all(quasiQ >= 0):
        raise ControlledError('/field_to_quasiprob/ quasiQ is not non-negative: quasiQ = %s' % quasiQ)

    # Return quasi probability distribution
    return quasiQ


# Convert field to normalized probability distribution
def field_to_prob(raw_phi): 
    phi = np.copy(raw_phi) 
    G = len(phi)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/field_to_prob/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/field_to_prob/ phi is not finite: phi = %s' % phi)

    # Re-level phi. NOTE: CHANGES PHI!
    phi -= min(phi)

    # Compute Q
    denom = sp.sum(sp.exp(-phi))
    Q = sp.exp(-phi)/denom

    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/field_to_prob/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/field_to_prob/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/field_to_prob/ Q is not non-negative: Q = %s' % Q)

    # Return probability
    return Q


# Convert probability distribution to field
def prob_to_field(Q):
    G = len(Q)

    # Make sure Q is valid
    if not all(np.isreal(Q)):
        raise ControlledError('/prob_to_field/ Q is not real: Q = %s' % Q)
    if not all(np.isfinite(Q)):
        raise ControlledError('/prob_to_field/ Q is not finite: Q = %s' % Q)
    if not all(Q >= 0):
        raise ControlledError('/prob_to_field/ Q is not non-negative: Q = %s' % Q)
    
    phi = -sp.log(G*Q + TINY_FLOAT64)

    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/prob_to_field/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/prob_to_field/ phi is not finite: phi = %s' % phi)

    # Return field
    return phi


def grid_info_from_bin_centers_1d(bin_centers):
    bin_centers = np.array(bin_centers)
    h = bin_centers[1]-bin_centers[0]
    bbox = [bin_centers[0]-h/2., bin_centers[-1]+h/2.]
    G = len(bin_centers)
    bin_edges = np.zeros(G+1)
    bin_edges[0] = bbox[0]
    bin_edges[-1] = bbox[1]
    bin_edges[1:-1] = bin_centers[:-1]+h/2.
    return bbox, h, bin_edges


def grid_info_from_bin_edges_1d(bin_edges):
    bin_edges = np.array(bin_edges)
    h = bin_edges[1]-bin_edges[1]
    bbox = [bin_edges[0], bin_edges[-1]]
    bin_centers = bin_edges[:-1]+h/2.
    return bbox, h, bin_centers


def grid_info_from_bbox_and_G(bbox, G):
    bin_edges = np.linspace(bbox[0], bbox[1], num=G+1, endpoint=True)
    h = bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges[:-1]+h/2.

    return h, bin_centers, bin_edges


# Make a 1d histogram. Bounding box is optional
def histogram_counts_1d(data, G, bbox, normalized=False):

    # Make sure normalized is valid
    if not isinstance(normalized, bool):
        raise ControlledError('/histogram_counts_1d/ normalized must be a boolean: normalized = %s' % type(normalized))

    # data_spread = max(data) - min(data)
    #
    # # Set lower bound automatically if called for
    # if bbox[0] == -np.Inf:
    #     bbox[0] = min(data) - data_spread*0.2
    #
    # # Set upper bound automatically if called for
    # if bbox[1] == np.Inf:
    #     bbox[1] = max(data) + data_spread*0.2

    # Crop data to bounding box
    indices = (data >= bbox[0]) & (data < bbox[1])
    cropped_data = data[0]

    # Get grid info from bbox and G
    h, bin_centers, bin_edges = grid_info_from_bbox_and_G(bbox, G)

    # Make sure h is valid
    if not (h > 0):
        raise ControlledError('/histogram_counts_1d/ h must be > 0: h = %s' % h)
    # Make sure bin_centers is valid
    if not (len(bin_centers) == G):
        raise ControlledError('/histogram_counts_1d/ bin_centers must have length %d: len(bin_centers) = %d' %
                              (G,len(bin_centers)))
    # Make sure bin_edges is valid
    if not (len(bin_edges) == G+1):
        raise ControlledError('/histogram_counts_1d/ bin_edges must have length %d: len(bin_edges) = %d' %
                              (G+1,len(bin_edges)))

    # Get counts in each bin
    counts, _ = np.histogram(data, bins=bin_edges, density=False)

    # Make sure counts is valid
    if not (len(counts) == G):
        raise ControlledError('/histogram_counts_1d/ counts must have length %d: len(counts) = %d' % (G, len(counts)))
    if not all(counts >= 0):
        raise ControlledError('/histogram_counts_1d/ counts is not non-negative: counts = %s' % counts)
    
    if normalized:
        hist = 1.*counts/np.sum(h*counts)
    else:
        hist = counts

    # Return the number of counts and the bin centers
    return hist, bin_centers


# Make a 2d histogram
def histogram_2d(data, box, num_bins=[10,10], normalized=False):
    data_x = data[0]
    data_y = data[1]

    hx, xs, x_edges = \
        grid_info_from_bbox_and_G(box[0], num_bins[0])
    hy, ys, y_edges = \
        grid_info_from_bbox_and_G(box[1], num_bins[1])

    hist, xedges, yedges = np.histogram2d(data_x, data_y, 
        bins=[x_edges, y_edges], normed=normalized)

    return hist, xs, ys


# Returns the left edges of a binning given the centers
def left_edges_from_centers(centers):
    h = centers[1]-centers[0]
    return centers - h/2.


# Returns the domain of a binning given the centers
def bounding_box_from_centers(centers):
    h = centers[1]-centers[0]
    xmin = centers[0]-h/2.
    xmax = centers[-1]+h/2.
    return sp.array([xmin,xmax])


# Defines a dot product with my normalization
def dot(v1,v2,h=1.0):
    v1r = v1.ravel()
    v2r = v2.ravel()
    G = len(v1)
    if not (len(v2) == G):
        raise ControlledError('/dot/ vectors are not of the same length: len(v1) = %d, len(v2) = %d' % (len(v1r), len(v2r)))
    return sp.sum(v1r*v2r*h/(1.*G))


# Computes a norm with my normalization
def norm(v,h=1.0):
    v_cc = np.conj(v)
    return sp.sqrt(dot(v,v_cc,h))


# Normalizes vectors (stored as columns of a 2D numpy array)
def normalize(vectors, grid_spacing=1.0):
    """ Normalizes vectors stored as columns of a 2D numpy array """
    G = vectors.shape[0] # length of each vector
    K = vectors.shape[1] # number of vectors

    # Set volume element h. This takes a little consideration
    if isinstance(grid_spacing,NUMBER):
        h = grid_spacing
    elif isinstance(grid_spacing,ARRAY):
        grid_spacing = sp.array(grid_spacing)
        h = sp.prod(grid_spacing)
    else:
        raise ControlledError('/normalize/ Cannot recognize h: h = %s' % h)
    
    if not (h > 0):
        raise ControlledError('/normalize/ h is not positive: h = %s' % h)

    norm_vectors = sp.zeros([G,K])
    for i in range(K):
        # Extract v from list of vectors
        v = vectors[:,i]
        # Flip sign of v so that last element is non-negative
        if (v[-1] < 0):
            v = -v
        # Normalize v and save in norm_vectors
        norm_vectors[:,i] = v/norm(v)

    # Return array with normalized vectors along the columns
    return norm_vectors


# Construct an orthonormal basis of order alpha from 1d legendre polynomials
def legendre_basis_1d(G, alpha, grid_spacing):

    # Create grid of centred x-values ranging from -1 to 1
    x_grid = (sp.arange(G) - (G-1)/2.)/(G/2.)

    # First create an orthogonal (not necessarily normalized) basis
    raw_basis = sp.zeros([G,alpha])
    for i in range(alpha):
        c = sp.zeros(alpha)
        c[i] = 1.0
        raw_basis[:,i] = legval(x_grid,c)

    # Normalize basis
    basis = normalize(raw_basis, grid_spacing)

    # Return normalized basis
    return basis


# Construct an orthonormal basis of order alpha from 2d legendre polynomials
def legendre_basis_2d(Gx, Gy, alpha, grid_spacing=[1.0,1.0]):

    # Compute x-coords and y-coords, each ranging from -1 to 1
    x_grid = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)

    # Create meshgrid of these
    xs, ys = np.meshgrid(x_grid,y_grid)
    basis_dim = alpha*(alpha+1)/2
    G = Gx*Gy
    raw_basis = sp.zeros([G,basis_dim])
    k = 0
    for a in range(alpha):
        for b in range(alpha):
            if a+b < alpha:
                c = sp.zeros([alpha,alpha])
                c[a,b] = 1
                raw_basis[:,k] = \
                    legval2d(xs,ys,c).T.reshape([G])
                k += 1

    # Normalize this basis using my convension
    basis = normalize(raw_basis, grid_spacing)
    
    # Return normalized basis
    return basis

def clean_numerical_input(x):
    """
    Returns a 1D np.array containing the numerical values in x or the
    value of x itself. Also returns well as a flag indicating whether x was
    passed as a single number or a list-like array.

    parameters
    ----------

    x: (number or list-like collection of numbers)
        The locations in the data domain at which to evaluate sampled
        density.

    returns
    -------

    x_arr: (1D np.array)
        Array containing numerical values of x.

    is_number: (bool)
        Flag indicating whether x was passed as a single number.

    """

    # If x is a number, record this fact and transform to np.array
    is_number = False
    if isinstance(x, numbers.Real):

        # Record that x is a single number
        is_number = True

        # Cast as 1D np.array of floats
        x = np.array([x]).astype(float)

    # Otherwise, if list-like, cast as 1D np.ndarray
    elif isinstance(x, LISTLIKE):

        # Cast as np.array
        x = np.array(x).ravel()

        # Check if x has any content
        check(len(x) > 0, 'x is empty.')

        # Check that entries are numbers
        check(all([isinstance(n, numbers.Real) for n in x]),
              'not all entries in x are real numbers')

        # Cast as 1D np.array of floats
        x = x.astype(float)

    # Otherwise, raise error
    else:
        raise ControlledError(
            'x is not a number or list-like, i.e., one of %s.'
            % str(LISTLIKE))

    # Make sure elements of x are all finite
    check(all(np.isfinite(x)),
          'Not all elements of x are finite.')

    return x, is_number


def check(condition, message):
    '''
    Checks a condition; raises a ControlledError with message if condition fails.
    :param condition:
    :param message:
    :return: None
    '''
    if not condition:
        raise ControlledError(message)


def enable_graphics(backend='TkAgg'):
    """
    Enable graphical output by suftware.

    This function should be _run before any calls are made to DensityEstimator.plot().
    This is not always necessary, since DensityEstimator.plot() itself will call this
    function if necessary. However, when plotting inline using the iPython
    notebook, this function must be called before the magic function
    ``%matplotlib inline``, e.g.::

        import suftware as sw
        sw.enable_graphics()
        %matplotlib inline

    If this function is never called, suftware can be _run without importing
    matplotlib. This can be useful, for instance, when distributing jobs
    across the nodes of a high performance computing cluster.


    parameters
    ----------

        backend: (str)
            Graphical backend to be passed to matplotlib.use().
            See the `matplotlib documentation <https://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`_
            for more information on graphical backends.


    returns
    -------

        None.

    """
    try:
        global mpl
        import matplotlib as mpl
        mpl.use(backend)
        global plt
        import matplotlib.pyplot as plt
    except:
        raise ControlledError('Could not import matplotlib.')


def handle_errors(func):
    """
    Decorator function to handle SUFTware errors
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Get should_fail debug flag
        should_fail = kwargs.pop('should_fail', None)

        try:

            # Execute function
            result = func(*args, **kwargs)
            error = False

            # If function didn't raise error, process results
            if should_fail is True:
                print('MISTAKE: Succeeded but should have failed.')
                mistake = True

            elif should_fail is False:
                print('Success, as expected.')
                mistake = False

            elif should_fail is None:
                mistake = False

            else:
                print('FATAL: should_fail = %s is not bool or None' %
                      should_fail)
                sys.exit(1)

        except ControlledError as e:
            error = True

            if should_fail is True:
                print('Error, as expected: ', e)
                mistake = False

            elif should_fail is False:
                print('MISTAKE: Failed but should have succeeded: ', e)
                mistake = True

            # Otherwise, just print an error and don't return anything
            else:
                print('Error: ', e)

        # If not in debug mode
        if should_fail is None:

            # If error, exit
            if error:
                sys.exit(1)

            # Otherwise, just return normal result
            else:
                return result

        # Otherwise, if in debug mode
        else:

            # If this is a constructor, set 'mistake' attribute of self
            if func.__name__ == '__init__':
                assert len(args) > 0
                args[0].mistake = mistake
                return None

            # Otherwise, create dummy object with mistake attribute
            else:
                obj = Dummy()
                obj.mistake = mistake
                return obj

    return wrapped_func

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# List of supported distributions by name
VALID_DISTRIBUTIONS = '''
gaussian
narrow
wide
foothills
accordian
goalposts
towers
uniform
beta_convex
beta_concave
exponential
gamma
triangular
laplace
vonmises
'''.split()

# Maximum number of samples this algorithm will simulate
MAX_DATASET_SIZE = 1E6

class Results(): pass;

def gaussian_mixture(N,weights,mus,sigmas,bbox):
    assert bbox[1] > bbox[0]
    assert len(weights)==len(mus)==len(sigmas)

    # Get xs to sample
    xs = np.linspace(bbox[0], bbox[1], 1E4)

    # Build pdf strings
    pdf_py = '0'
    pdf_js = '0'
    for m, s, w in zip(mus,sigmas,weights):
        pdf_py +='+(%f/%f)*np.exp(-0.5*np.power((x-(%f))/%f,2))'%(w,s,m,s)
        pdf_js +='+(%f/%f)*Math.exp(-0.5*Math.pow((x-(%f))/%f,2))'%(w,s,m,s)

    # Evaluate pdf at xs
    ps = np.zeros(len(xs))
    for i,x in enumerate(xs):
        ps[i] = eval(pdf_py)
    ps /= sum(ps)

    # Sample datapoints
    data = np.random.choice(xs, size=N, replace=True, p=ps)

    # Return valuables
    return data, pdf_py, pdf_js

class SimulatedDataset:
    """
    Simulates data from a variety of distributions.

    parameters
    ----------

    distribution: (str)
        The distribution from which to draw data. Run sw.SimulatedDataset.list()
        to which distributions are available.

    num_data_points: (int > 0)
        The number of data points to simulate. Must satisfy
        0 <= N <= MAX_DATASET_SIZE.

    seed: (int)
        Seed passed to random number generator.

    attributes
    ----------

    data: (np.array)
        The simulated dataset

    bounding_box: ([float, float])
        Bounding box within which data is generated.

    distribution: (str)
        Name of the simualted distribution

    pdf_js: (str)
        Formula for probability density in JavaScript

    pdf_py: (str)
        Formula for probaiblity density in Python.

    periodic: (bool)
        Whether the simulated distribution is periodic within bounding_box.

    """

    @handle_errors
    def __init__(self,
                 distribution='gaussian',
                 num_data_points=100,
                 seed=None):

        # Check that distribution is valid
        check(distribution in self.list(),
              'distribution = %s is not valid' % distribution)

        # Check num_data_points is integral
        check(isinstance(num_data_points, numbers.Integral),
              'num_data_points = %s is not an integer.' % num_data_points)

        # Cast num_data_points as an integer
        num_data_points = int(num_data_points)

        # Check value
        check(0 < num_data_points <= MAX_DATASET_SIZE,
              'num_data_points = %d; must have 0 < num_data_points <= %d.'
              % (num_data_points, MAX_DATASET_SIZE))

        # Run seed and catch errors
        try:
            np.random.seed(seed)
        except TypeError:
            raise ControlledError('type(seed) = %s; invalid type.' % type(seed))
        except ValueError:
            raise ControlledError('seed = %s; invalid value.' % seed)

        # Set default value for periodic
        periodic = False

        # If gaussian distribution
        if distribution == 'gaussian':
            description = 'Gaussian distribution'
            mus = [0.]
            sigmas = [1.]
            weights = [1.]
            bounding_box = [-5,5]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        # If mixture of two gaussian distributions
        elif distribution == 'narrow':
            description = 'Gaussian mixture, narrow separation'
            mus = [-1.25, 1.25]
            sigmas = [1., 1.]
            weights = [1., 1.]
            bounding_box = [-6, 6]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        # If mixture of two gaussian distributions
        elif distribution == 'wide':
            description = 'Gaussian mixture, wide separation'
            mus = [-2.0, 2.0]
            sigmas = [1.0, 1.0]
            weights = [1.0, 0.5]
            bounding_box = [-6.0, 6.0]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        elif distribution == 'foothills':
            description = 'Foothills (Gaussian mixture)'
            mus = [0., 5., 8., 10, 11]
            sigmas = [2., 1., 0.5, 0.25, 0.125]
            weights = [1., 1., 1., 1., 1.]
            bounding_box = [-5,12]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        elif distribution == 'accordian':
            description = 'Accordian (Gaussian mixture)'
            mus = [0., 5., 8., 10, 11, 11.5]
            sigmas = [2., 1., 0.5, 0.25, 0.125, 0.0625]
            weights = [16., 8., 4., 2., 1., 0.5]
            bounding_box = [-5,13]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        elif distribution == 'goalposts':
            description = 'Goalposts (Gaussian mixture)'
            mus = [-20, 20]
            sigmas = [1., 1.]
            weights = [1., 1.]
            bounding_box = [-25,25]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        elif distribution == 'towers':
            description = 'Towers (Gaussian mixture)'
            mus = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
            sigmas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
            bounding_box = [-25,25]
            data, pdf_py, pdf_js = gaussian_mixture(num_data_points, weights, mus, sigmas, bounding_box)

        # If uniform distribution
        elif distribution == 'uniform':
            data = stats.uniform.rvs(size=num_data_points)
            bounding_box = [0,1]
            description = 'Uniform distribution'
            pdf_js = "1.0"
            pdf_py = "1.0"

        # Convex beta distribution
        elif distribution == 'beta_convex':
            data = stats.beta.rvs(a=0.5, b=0.5, size=num_data_points)
            bounding_box = [0,1]
            description = 'Convex beta distribtuion'
            pdf_js = "Math.pow(x,-0.5)*Math.pow(1-x,-0.5)*math.gamma(1)/(math.gamma(0.5)*math.gamma(0.5))"
            pdf_py = "np.power(x,-0.5)*np.power(1-x,-0.5)*math.gamma(1)/(math.gamma(0.5)*math.gamma(0.5))"

        # Concave beta distribution
        elif distribution == 'beta_concave':
            data = stats.beta.rvs(a=2, b=2, size=num_data_points)
            bounding_box = [0,1]
            description = 'Concave beta distribution'
            pdf_js = "Math.pow(x,1)*Math.pow(1-x,1)*math.gamma(4)/(math.gamma(2)*math.gamma(2))"
            pdf_py = "np.power(x,1)*np.power(1-x,1)*math.gamma(4)/(math.gamma(2)*math.gamma(2))"

        # Exponential distribution
        elif distribution == 'exponential':
            data = stats.expon.rvs(size=num_data_points)
            bounding_box = [0,5]
            description = 'Exponential distribution'
            pdf_js = "Math.exp(-x)"
            pdf_py = "np.exp(-x)"

        # Gamma distribution
        elif distribution == 'gamma':
            data = stats.gamma.rvs(a=3, size=num_data_points)
            bounding_box = [0,10]
            description = 'Gamma distribution'
            pdf_js = "Math.pow(x,2)*Math.exp(-x)/math.gamma(3)"
            pdf_py = "np.power(x,2)*np.exp(-x)/math.gamma(3)"

        # Triangular distribution
        elif distribution == 'triangular':
            data = stats.triang.rvs(c=0.5, size=num_data_points)
            bounding_box = [0,1]
            description = 'Triangular distribution'
            pdf_js = "2-4*Math.abs(x - 0.5)"
            pdf_py = "2-4*np.abs(x - 0.5)"

        # Laplace distribution
        elif distribution == 'laplace':
            data = stats.laplace.rvs(size=num_data_points)
            bounding_box = [-5,5]
            description = "Laplace distribution"
            pdf_js = "0.5*Math.exp(- Math.abs(x))"
            pdf_py = "0.5*np.exp(- np.abs(x))"

        # von Misses distribution
        elif distribution == 'vonmises':
            data = stats.vonmises.rvs(1, size=num_data_points)
            bounding_box = [-3.14159,3.14159]
            periodic = True
            description = 'von Mises distribution'
            pdf_js = "Math.exp(Math.cos(x))/7.95493"
            pdf_py = "np.exp(np.cos(x))/7.95493"

        else:
            raise ControlledError('Distribution type "%s" not recognized.' % distribution)

        # Set these
        attributes = {
            'data': data,
            'bounding_box': bounding_box,
            'distribution': distribution,
            'pdf_js': pdf_js,
            'pdf_py': pdf_py,
            'periodic': periodic
        }
        for key, value in attributes.items():
            setattr(self, key, value)

    @staticmethod
    @handle_errors
    def list():
        """
        Return list of valid distributions.
        """

        return VALID_DISTRIBUTIONS


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#$$$$ ---- ścieżki ----

# Load directory of file
data_dir = os.path.dirname(os.path.abspath(__file__))+'/../examples/data'

# List of supported distributions by name
VALID_DATASETS = ['.'.join(name.split('.')[:-1]) for name in
                  os.listdir(data_dir) if '.txt' in name]
VALID_DATASETS.sort()

class ExampleDataset:
    """
    Provides an interface to example data provided with the SUFTware package.

    parameters
    ----------

    dataset: (str)
        Name of dataset to load. Run sw.ExampleDataset.list() to see
        which datasets are available.

    attributes
    ----------

    data: (np.array)
        An array containing sampled data

    details: (np.array, optional)
        Optional return value containing meta information


    """

    # Constructor
    @handle_errors
    def __init__(self, dataset='old_faithful_eruption_times'):

        # Check that dataset is valid
        check(dataset in self.list(),
              'Distribution "%s" not recognized.' % dataset)

        # Set file dataset
        file_name = '%s/%s.txt' % (data_dir, dataset)

        # Load data
        self._load_dataset(file_name)

    @handle_errors
    def _load_dataset(self, file_name):
        # Load data
        self.data = np.genfromtxt(file_name)

        # Fill in details from data file header
        details = {}
        header_lines = [line.strip()[1:] for line in open(file_name, 'r')
                        if line.strip()[0] == '#']
        for line in header_lines:
            key = eval(line.split(':')[0])
            value = eval(line.split(':')[1])
            try:
                setattr(self, key, value)
            except:
                ControlledError('Error loading example data. Either key or value'
                          'of metadata is invalid. key = %s, value = %s' %
                                (key, value))

    @staticmethod
    @handle_errors
    def list():
        """
        Return list of available datasets.
        """
        return VALID_DATASETS

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Class container for Laplacian operators. Constructor computes spectrum.
class Laplacian:
    """
    Class container for Laplacian operators. Constructor computes specturm.

    Methods:
        get_G(): 
            Returns the (total) number of gridpoints

        get_kernel_dim(): 
            Returns the dimension of the kernel

        get_dense_matrix(): 
            Returns a dense scipy matrix of the operator

        get_sparse_matrix(): 
            Returns a scipy.sparse csr matrix of the operator

        save_to_file(filename):
            Pickles instance of class and saves to disk.
    """

    def __init__(self, operator_type, operator_order, num_gridpoints, grid_spacing=1.0 ):
        """
        Constructor for Smoothness_operator class

        Args:
            operator_type (str): 
                The type of operator. Accepts one of the following values:
                    '1d_bilateral'
                    '1d_periodic'
                    '2d_bilateral'
                    '2d_periodic'

            operator_order (int): 
                The order of the operator.

            num_gridpoints: 
                The number of gridpoints in each dimension of the domain.
        """
        
        # Make sure grid_spacing is valid
        if not isinstance(grid_spacing, float):
            raise ControlledError('/Laplacian/ grid_spacing must be a float: grid_spacing = %s' % type(grid_spacing))
        if not (grid_spacing > 0):
            raise ControlledError('/Laplacian/ grid_spacing must be > 0: grid_spacing = %s' % grid_spacing)
        
        if '1d' in operator_type:
            self._coordinate_dim = 1

            # Make sure operator_type is valid
            if operator_type == '1d_bilateral':
                periodic = False
            elif operator_type == '1d_periodic':
                periodic = True
            else:
                raise ControlledError('/Laplacian/ Cannot identify operator_type: operator_type = %s' % operator_type)
                
            self._type = operator_type
            
            self._sparse_matrix, self._kernel_basis = \
                laplacian_1d(num_gridpoints, operator_order, grid_spacing, periodic)
            
            self._G = self._kernel_basis.shape[0]
            self._kernel_dim = self._kernel_basis.shape[1]
            self._alpha = operator_order

        elif '2d' in operator_type:
            self._coordinate_dim = 2

            assert( len(num_gridpoints)==2 )
            assert( all([isinstance(n,utils.NUMBER) for n in num_gridpoints]) )

            assert( len(grid_spacing)==2 )
            assert( all([isinstance(n,utils.NUMBER) for n in grid_spacing]) )

            if operator_type == '2d_bilateral':
                periodic = False
            elif operator_type == '2d_periodic':
                periodic = True
            else:
                raise ControlledError('ERROR: cannot identify operator_type.')

            
            self._type = operator_type
            
            self._sparse_matrix, self._kernel_basis = \
                laplacian_2d( num_gridpoints, 
                              operator_order, 
                              grid_spacing, 
                              periodic=periodic, 
                              sparse=True,
                              report_kernel=True)

            self._Gx = int(num_gridpoints[0])
            self._Gy = int(num_gridpoints[1])
            self._G = self._Gx * self._Gy
            self._alpha = operator_order
            assert( self._G == self._kernel_basis.shape[0] )
            self._kernel_dim = self._kernel_basis.shape[1]

        else:
            raise ControlledError('/Laplacian/ Cannot identify operator_type: operator_type = %s' % operator_type)

        # Compute spectrum, and set lowest rank eigenvectors as kernel
        self._dense_matrix = self._sparse_matrix.todense()
        eigenvalues, eigenvectors = eigh(self._dense_matrix)
        self._eigenvalues = eigenvalues
        self._eigenbasis = utils.normalize(eigenvectors)
        #self._kernel_basis = self._eigenbasis[:,:self._kernel_dim]

        # Set kernel eigenvalues and eigenvectors
        self._eigenvalues[:self._kernel_dim] = 0.0
        self._eigenbasis[:,:self._kernel_dim] = self._kernel_basis

    def get_G(self):
        """ Return the total number of gridpoints used by this operator. """
        return self._G
    
    def get_kernel_basis(self):
        """ Returns the kernel as a kernel_dim x G numpy array """
        return sp.copy(self._kernel_basis)

    def get_kernel_dim(self):
        """ Return the dimension of the kernel of this operator. """
        return self._kernel_dim
    
    def get_sparse_matrix(self):
        """ Return a sparse matrix version of this operator. """
        return self._sparse_matrix

    def get_sparse_Lambda(self):
        """ Return a sparse matrix version of Lambda. """
        return self._sparse_matrix

    def get_dense_matrix(self):
        """ Return a dense matrix version of this operator. """
        return self._sparse_matrix.todense()

    def get_dense_Lambda(self):
        """ Return a dense matrix version of Lambda. """
        return self._sparse_matrix.todense()

    # def save(self, filename):
    #     """ Saves the current Laplacian in a way that can be recovered """
    #     pickle.dump(self, file(filename, 'w'))


# # Function for loading Laplacian from file
# def load(filename):
#     """ Loads a picked Laplacian from a file, and returns instance. """
#     operator = pickle.load(file(filename))
#     return operator


def derivative_matrix_1d(G, grid_spacing):
    """ Returns a (G-1) x G sized 1d derivative matrix. """
    
    # Create matrix
    tmp_mat = sp.diag(sp.ones(G),0) + sp.diag(-1.0*sp.ones(G-1),-1)
    right_partial = tmp_mat[1:,:]/grid_spacing
    
    return sp.mat(right_partial)


def laplacian_1d(G, alpha, grid_spacing, periodic, sparse=True, report_kernel=True):
    """ Returns a G x G sized 1d bilateral laplacian matrix of order alpha """

    # Make sure sparse is valid
    if not isinstance(sparse, bool):
        raise ControlledError('/laplacian_1d/ sparse must be a boolean: sparse = %s' % type(sparse))
    # Make sure report_kernel is valid
    if not isinstance(report_kernel, bool):
        raise ControlledError('/laplacian_1d/ report_kernel must be a boolean: report_kernel = %s' % type(report_kernel))
    
    x_grid = (sp.arange(G) - (G-1)/2.)/(G/2.)

    # If periodic boundary conditions, construct regular laplacian
    if periodic:
        tmp_mat = 2*sp.diag(sp.ones(G),0) - sp.diag(sp.ones(G-1),-1) - sp.diag(sp.ones(G-1),+1)
        tmp_mat[G-1,0] = -1.0
        tmp_mat[0,G-1] = -1.0
        Delta = (sp.mat(tmp_mat)/(grid_spacing**2))**alpha
        
        # Get kernel, which is just the constant vector v = sp.ones([G,1])
        # kernel_basis = utils.normalize(v, grid_spacing)
        kernel_basis = utils.legendre_basis_1d(G, 1, grid_spacing)

    # Otherwise, construct bilateral laplacian
    else:
    
        # Initialize to G x G identity matrix
        right_side = sp.diag(sp.ones(G),0)
        
        # Multiply alpha derivative matrices of together. Reduce dimension going left
        for a in range(alpha):
            right_side = derivative_matrix_1d(G-a, grid_spacing)*right_side 
        
        # Construct final bilateral laplacian
        Delta = right_side.T*right_side

        # Make sure Delta is valid
        if not (Delta.shape[0] == Delta.shape[1] == G):
            raise ControlledError('/laplacian_1d/ Delta must have shape (%d, %d): Delta.shape = %s' % (G, G, Delta.shape))

        # Construct a basis for the kernel from legendre polynomials
        kernel_basis = utils.legendre_basis_1d(G, alpha, grid_spacing)

        # Make sure kernel_basis is valid
        if not ((kernel_basis.shape[0] == G) and (kernel_basis.shape[1] == alpha)):
            raise ControlledError('/laplacian_1d/ kernel_basis must have shape (%d, %d): kernel_basis.shape = %s' %
                                  (G,alpha,kernel_basis.shape))
        
    # Sparsify matrix if requested
    if sparse:  
        Delta = csr_matrix(Delta)

    # Report kernel if requested
    if report_kernel:
        return Delta, kernel_basis

    # Otherwise, just report matrix
    else:
        return Delta


def laplacian_2d( num_gridpoints, alpha, grid_spacing=[1.0,1.0], periodic=False, sparse=False, report_kernel=False):
    """ Returns a GxG (G=GxGy) sized 2d Laplacian """
    assert(len(num_gridpoints)==2)
    Gx = num_gridpoints[0]
    Gy = num_gridpoints[1]
    G = Gx*Gy
    assert(Gx == int(Gx))
    assert(Gy == int(Gy))
    assert(alpha == int(alpha))
    assert(alpha >= 1)
    assert(len(grid_spacing)==2)
    assert(type(grid_spacing[0]) == float)
    assert(type(grid_spacing[1]) == float)
    hx = grid_spacing[0]
    hy = grid_spacing[0]
    assert(hx > 0.0)
    assert(hy > 0.0)
    
    # Identity matrices, which will be used below
    I_x = sp.mat(sp.identity(Gx))
    I_y = sp.mat(sp.identity(Gy))

    # Compute x-coords and y-coords
    x_grid = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)
    xs,ys = np.meshgrid(x_grid,y_grid)
    
    # If periodic boundary conditions,
    if periodic:
        Delta_x = laplacian_1d(Gx, alpha=1, grid_spacing=hx, periodic=True)    
        Delta_y = laplacian_1d(Gy, alpha=1, grid_spacing=hy, periodic=True)
        
        # Use the kroneker product to generate a first-order operator
        Delta_1 = sp.mat(sp.kron(Delta_x,I_y) + sp.kron(I_x,Delta_y))

        # Raise operator to alpha power 
        Delta = Delta_1**alpha
    
    # If bilateral, construct alpha-order bilateral laplacian algorithmically
    else:
        Delta_x_array = [I_x]
        Delta_y_array = [I_y]

        for a in range(1,alpha+1):
            Delta_x_array.append( laplacian_1d(Gx, alpha=a, grid_spacing=hx) )
            Delta_y_array.append( laplacian_1d(Gy, alpha=a, grid_spacing=hy) )

        for a in range(alpha+1):
            Dx = Delta_x_array[alpha-a]
            Dy = Delta_y_array[a]
            coeff = comb(alpha,a)
            if a == 0:
                Delta = coeff*sp.mat(sp.kron(Dx,Dy))
            else:
                Delta += coeff*sp.mat(sp.kron(Dx,Dy))
        
    # Build kernel from 2d legendre polynomials
    if periodic:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, 1, grid_spacing)
    else:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, alpha, grid_spacing)

    # Sparsify matrix if requested
    if sparse:  
        Delta = csr_matrix(Delta)

    # Report kernel if requested
    if report_kernel:
        return Delta, kernel_basis

    # Otherwise, just report matrix
    else:
        return Delta

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

x_MIN = -500


# Laplace approach with importance sampling
def Laplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel,
                     pt_sampling=False):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = np.abs(sp.array(U_mat_on_kernel[0]))
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        # H = deft_code.deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = np.abs(sp.array(U_mat[0]))
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N / G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), eig_vals)
        outputs = pool.map(y_sampling_of_Lap, inputs)
        y_samples = sp.array(outputs)
    else:
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [num_samples, eig_vals[i]]
            outputs = y_sampling_of_Lap(inputs)
            y_samples[i, :] = outputs

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i, :]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G, num_samples])
    for k in range(num_samples):
        phi_samples[:, k] = x_samples[:, k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones(
        [G, num_samples]) + x_samples - 0.5 * np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    phi_weights = sp.exp(-dS_vals)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean and sample mean std
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    correction = sp.log(w_sample_mean)
    return correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, draw y samples according to the distribution
def y_sampling_of_Lap(input_array):
    num_samples = input_array[0]
    eig_val = input_array[1]

    # Draw y samples
    sigma = 1.0 / sp.sqrt(eig_val)
    y_samples = np.random.normal(0, sigma, num_samples)

    # Return y samples
    return y_samples


# Generalized Laplace approach with importance sampling
def GLaplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel,
                      sampling=True, pt_sampling=False, num_grid=400):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = np.abs(sp.array(U_mat_on_kernel[0]))
        eig_vecs = sp.array((sp.mat(kernel_basis) * U_mat_on_kernel[1]).T)
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = sp.array(U_mat[1].T)
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N / G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, calculate gamma and draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples),
                                itertools.repeat(num_grid),
                                eig_vals, eig_vecs, itertools.repeat(lambdas),
                                itertools.repeat(sampling))
        outputs = pool.map(y_sampling_of_GLap, inputs)
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            gammas[i] = outputs[i][0]
            if sampling:
                y_samples[i, :] = outputs[i][1]
    else:
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [num_samples, num_grid, eig_vals[i], eig_vecs[i, :],
                      lambdas, sampling]
            outputs = y_sampling_of_GLap(inputs)
            gammas[i] = outputs[0]
            if sampling:
                y_samples[i, :] = outputs[1]

    # If not sampling, return correction and other stuff at this point
    if not sampling:
        correction = sp.sum(sp.log(gammas))
        w_sample_mean = 1.0
        w_sample_mean_std = 0.0
        return correction, w_sample_mean, w_sample_mean_std

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i, :]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G, num_samples])
    for k in range(num_samples):
        phi_samples[:, k] = x_samples[:, k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones(
        [G, num_samples]) + x_samples - 0.5 * np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    if go_parallel:
        inputs = itertools.izip(sp.array(transf_matrix.T), y_samples,
                                itertools.repeat(lambdas))
        outputs = pool.map(dSi_evaluations_of_GLap, inputs)
        dSi_vals = sp.array(outputs)
    else:
        dSi_vals = np.zeros([y_dim, num_samples])
        for i in range(y_dim):
            inputs = [sp.array(transf_matrix)[:, i], y_samples[i, :], lambdas]
            outputs = dSi_evaluations_of_GLap(inputs)
            dSi_vals[i, :] = outputs
    sum_dSi_vals = sp.array(sp.mat(np.ones(y_dim)) * sp.mat(dSi_vals)).ravel()
    dS_residues = dS_vals - sum_dSi_vals
    dS_residues[dS_residues < x_MIN] = x_MIN
    phi_weights = sp.exp(-dS_residues)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean, sample mean std, and effective sample size of the weights
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    correction = sp.sum(sp.log(gammas)) + sp.log(w_sample_mean)
    return correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, calculate gamma and draw y samples according to the distribution
def y_sampling_of_GLap(inputs_array):
    num_samples = inputs_array[0]
    num_grid = inputs_array[1]
    eig_val = inputs_array[2]
    eig_vec = inputs_array[3]
    lambdas = inputs_array[4]
    sampling = inputs_array[5]

    # Find the lower and upper bounds of the Laplace distribution and tabulate its values
    sigma = 1.0 / sp.sqrt(eig_val)
    Lap_N_lb = 0
    while distribution(eig_val, eig_vec, Lap_N_lb * sigma, lambdas,
                       switch=0) > 1E-6:
        Lap_N_lb -= 1
    Lap_N_ub = 0
    while distribution(eig_val, eig_vec, Lap_N_ub * sigma, lambdas,
                       switch=0) > 1E-6:
        Lap_N_ub += 1
    Lap_Ns = []
    Lap_Es = []
    for Lap_N in range(Lap_N_lb, Lap_N_ub + 1):
        Lap_Ns.append(Lap_N)
        Lap_Es.append(
            distribution(eig_val, eig_vec, Lap_N * sigma, lambdas, switch=0))

    # Find the lower and upper bounds of the generalized Laplace distribution and tabulate its values
    sigma = 1.0 / sp.sqrt(eig_val)
    GLap_N_lb = 0
    while distribution(eig_val, eig_vec, GLap_N_lb * sigma, lambdas,
                       switch=1) > 1E-6:
        GLap_N_lb -= 1
    GLap_N_ub = 0
    while distribution(eig_val, eig_vec, GLap_N_ub * sigma, lambdas,
                       switch=1) > 1E-6:
        GLap_N_ub += 1
    GLap_Ns = []
    GLap_Es = []
    for GLap_N in range(GLap_N_lb, GLap_N_ub + 1):
        GLap_Ns.append(GLap_N)
        GLap_Es.append(
            distribution(eig_val, eig_vec, GLap_N * sigma, lambdas, switch=1))

    # See if these two distributions are similar enough
    if Lap_Ns == GLap_Ns:
        diff_Es = abs(sp.array(Lap_Es) - sp.array(GLap_Es))
        if all(diff_Es < 1E-6):
            similar_enough = True
        else:
            similar_enough = False
    else:
        similar_enough = False

    # If these two distributions are similar enough, set gamma to 1, otherwise do the integration
    if similar_enough:
        gamma = 1.0
    else:
        # Evaluate area under the Laplace distribution
        Lap_bin_edges = sp.linspace(Lap_Ns[0] * sigma, Lap_Ns[-1] * sigma,
                                    num_grid + 1)
        h = Lap_bin_edges[1] - Lap_bin_edges[0]
        Lap_bin_centers = sp.linspace(Lap_Ns[0] * sigma + h / 2,
                                      Lap_Ns[-1] * sigma - h / 2, num_grid)
        Lap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            Lap_bin_centers_dist[j] = distribution(eig_val, eig_vec,
                                                   Lap_bin_centers[j], lambdas,
                                                   switch=0)
        Lap_area = h * sp.sum(Lap_bin_centers_dist)
        # Evaluate area under the generalized Laplace distribution
        GLap_bin_edges = sp.linspace(GLap_Ns[0] * sigma, GLap_Ns[-1] * sigma,
                                     num_grid + 1)
        h = GLap_bin_edges[1] - GLap_bin_edges[0]
        GLap_bin_centers = sp.linspace(GLap_Ns[0] * sigma + h / 2,
                                       GLap_Ns[-1] * sigma - h / 2, num_grid)
        GLap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            GLap_bin_centers_dist[j] = distribution(eig_val, eig_vec,
                                                    GLap_bin_centers[j],
                                                    lambdas, switch=1)
        GLap_area = h * sp.sum(GLap_bin_centers_dist)
        # Take ratio of the areas
        gamma = GLap_area / Lap_area

    # If not sampling, return gamma at this point
    if not sampling:
        return [gamma]

    # If the distribution is normal, draw samples from it directly
    if similar_enough:
        y_samples = np.random.normal(0, sigma, num_samples)

    # Otherwise, draw samples according to the distribution as follows
    else:
        bin_edges = sp.linspace(GLap_N_lb * sigma, GLap_N_ub * sigma,
                                num_grid + 1)
        h = bin_edges[1] - bin_edges[0]
        bin_centers = sp.linspace(GLap_N_lb * sigma + h / 2,
                                  GLap_N_ub * sigma - h / 2, num_grid)
        bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            bin_centers_dist[j] = distribution(eig_val, eig_vec, bin_centers[j],
                                               lambdas, switch=1)
        prob = bin_centers_dist / sp.sum(bin_centers_dist)
        y_samples = np.random.choice(bin_centers, num_samples, replace=True,
                                     p=prob)
        y_shifts = (np.random.random(num_samples) - 0.5 * np.ones(
            num_samples)) * h
        y_samples += y_shifts
        """
            Below is a HOT spot !!!

            # Randomly distribute the samples within each bin
            indices = (y_samples-h/2-N_lb*sigma) / h
            for k in range(num_samples):
                index = int(indices[k])
                a = y_grid[index]
                fa = grid_dist[index]
                fb = grid_dist[index+1]
                r = np.random.rand()
                if fa == fb:
                    y_samples[k] = a + h * r
                else:
                    h_ratio = (sp.sqrt(fa**2+r*(fb**2-fa**2)) - fa) / (fb - fa)
                    y_samples[k] = a + h * h_ratio
            """

    # Return gamma and y samples
    return [gamma, y_samples]


# Evaluations of dSi
def dSi_evaluations_of_GLap(inputs_array):
    Ui = inputs_array[0]
    yi = inputs_array[1]
    lambdas = inputs_array[2]

    G = len(Ui)
    num_samples = len(yi)

    xi = sp.array(sp.mat(Ui).T * sp.mat(yi))
    for i in range(G):
        xi_vec = xi[i, :]
        xi_vec[xi_vec < x_MIN] = x_MIN
    xi_combo = sp.exp(-xi) - np.ones([G, num_samples]) + xi - 0.5 * np.square(
        xi)

    return sp.array(sp.mat(lambdas) * sp.mat(xi_combo)).ravel()


# The Laplace or generalized Laplace distribution
def distribution(eig_val, eig_vec, y, lambdas, switch):
    return sp.exp(
        -(0.5 * eig_val * y ** 2 + switch * dSi(eig_vec * y, lambdas)))


# The dSi function
def dSi(x, lambdas):
    x[x < x_MIN] = x_MIN
    return sp.sum(lambdas * (sp.exp(-x) - 1.0 + x - 0.5 * x ** 2))


# Feynman diagram calculations
def Feynman_diagrams(phi_t, R, Delta, t, N):
    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        # Evaluate propagator matrix
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N / G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        M_inv_on_kernel = sp.linalg.inv(M_mat_on_kernel)
        P_mat = sp.mat(kernel_basis) * M_inv_on_kernel * sp.mat(kernel_basis).T
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N / G)
    else:
        G = len(phi_t)
        # Evaluate propagator matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        P_mat = np.linalg.inv(A_mat)
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N / G)

    # Calculate Feynman diagrams
    correction = diagrams_1st_order(G, P_mat, V)

    # Return the correction and other stuff
    w_sample_mean = 1.0
    w_sample_mean_std = 0.0
    return correction, w_sample_mean, w_sample_mean_std


# Feynman diagrams of order 1/N
def diagrams_1st_order(G, P, V):
    s = np.zeros(4)  # s[0] is dummy

    P_diag = sp.array([P[i, i] for i in range(G)])

    # Diagram 1
    s[1] = sp.sum(V * P_diag ** 2)
    s[1] *= -1 / 8

    # Diagram 2
    U = sp.array([V[i] * P_diag[i] for i in range(G)])
    s[2] = sp.array(sp.mat(U) * P * sp.mat(U).T).ravel()[0]
    s[2] *= 1 / 8

    # Diagram 3
    s[3] = sp.array(sp.mat(V) * sp.mat(sp.array(P) ** 3) * sp.mat(V).T).ravel()[
        0]
    s[3] *= 1 / 12

    """
    # Diagram 1
    for i in range(G):
        s[1] += V[i] * P[i,i]**2
    s[1] *= -1/8

    # Diagram 2
    for i in range(G):
        for j in range(G):
            s[2] += V[i] * V[j] * P[i,i] * P[i,j] * P[j,j]
    s[2] *= 1/8

    # Diagram 3
    for i in range(G):
        for j in range(G):
            s[3] += V[i] * V[j] * P[i,j]**3
    s[3] *= 1/12
    """

    # Return
    return sp.sum(s)


# Feynman diagrams of order 1/N^2 ---> Under construction
def diagrams_2nd_order(G, P, V):
    time1 = time.time()
    num_samples = 300000  # Use different value for large and small t ?
    index_samples = np.random.randint(0, G - 1, 4 * num_samples).reshape(
        [num_samples, 4])

    s_array = np.zeros(num_samples)
    for n in range(num_samples):
        i = index_samples[n, 0]
        j = index_samples[n, 1]
        k = index_samples[n, 2]
        l = index_samples[n, 3]
        s_array[n] = V[i] * V[j] * V[k] * V[l] * P[i, j] * P[i, k] * P[i, l] * \
                     P[j, k] * P[j, l] * P[k, l]
    s = sp.sum(s_array) * G ** 4 / num_samples
    ms = sp.mean(s_array)
    ds = sp.std(s_array) / sp.sqrt(num_samples)
    print('')
    print('s =', s)
    print(ds / ms)
    print('time 1 =', time.time() - time1)

    """
    time2 = time.time()
    s2 = 0
    for i in range(G):
        for j in range(G):
            for k in range(G):
                for l in range(G):
                    s2 += V[i] * V[j] * V[k] * V[l] * P[i,j] * P[i,k] * P[i,l] * P[j,k] * P[j,l] * P[k,l]
    print 's2 =', s2
    print 'time 2 =', time.time()-time2
    """

    return 0


# Metropolis Monte Carlo
def Metropolis_Monte_Carlo(phi_t, R, Delta, t, N, num_samples, go_parallel,
                           pt_sampling):
    G = len(phi_t)
    num_thermalization_steps = 10 * G
    num_steps_per_sample = G

    phi_samples = np.zeros([G, num_samples])
    sample_index = 0

    # Prepare the stuff for the case of maxent or finite t, and then do Monte Carlo sampling
    if not np.isfinite(t):

        # Find the kernel basis
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N / G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G, alpha])
        for i in range(alpha):
            kernel_basis[:, i] = Delta_diagonalized[1][:, i].ravel()

        # Find coefficients of phi_t in the kernel basis
        coeffs = np.zeros(alpha)
        for i in range(alpha):
            coeffs[i] = sp.mat(kernel_basis[:, i]) * sp.mat(phi_t).T

        # Find eigen-modes of the Hessian matrix
        H = maxent.hessian_per_datum_from_coeffs(coeffs, R, kernel_basis)
        A_mat = sp.mat(H) * N
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = np.abs(sp.array(U_mat[1]))

        # Initialize
        coeffs_current = coeffs
        S_current = maxent.action_per_datum_from_coeffs(coeffs_current, R,
                                                        kernel_basis) * N

        # Do Monte Carlo sampling
        for k in range(
                                num_thermalization_steps + num_samples * num_steps_per_sample + 1):
            i = np.random.randint(0, alpha)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[i, :]
            step_size = np.random.normal(0, 1.0 / np.sqrt(eig_val))
            coeffs_new = coeffs_current + eig_vec * step_size
            S_new = maxent.action_per_datum_from_coeffs(coeffs_new, R,
                                                        kernel_basis) * N
            if np.log(np.random.uniform(0, 1)) < (S_current - S_new):
                coeffs_current = coeffs_new
                S_current = S_new
            if (k > num_thermalization_steps) and (
                    k % num_steps_per_sample == 0):
                phi_samples[:, sample_index] = maxent.coeffs_to_field(
                    coeffs_current, kernel_basis)
                sample_index += 1

    else:

        # Find eigen-modes of the Hessian matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N / G)
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = np.abs(sp.array(U_mat[0]))
        eig_vecs = np.abs(sp.array(U_mat[1]))

        # Initialize
        phi_current = phi_t
        S_current = deft_core.action(phi_current, R, Delta, t, N) * (N / G)

        # Do Monte Carlo sampling
        for k in range(
                                num_thermalization_steps + num_samples * num_steps_per_sample + 1):
            i = np.random.randint(0, G)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[:, i]
            step_size = np.random.normal(0, 1.0 / np.sqrt(eig_val))
            phi_new = phi_current + eig_vec * step_size
            S_new = deft_core.action(phi_new, R, Delta, t, N) * (N / G)
            if np.log(np.random.uniform(0, 1)) < (S_current - S_new):
                phi_current = phi_new
                S_current = S_new
            if (k > num_thermalization_steps) and (
                    k % num_steps_per_sample == 0):
                phi_samples[:, sample_index] = phi_current
                sample_index += 1

    # Return phi samples and phi weights
    return phi_samples, np.ones(num_samples)


# Sample probable densities using posterior probability
def posterior_sampling(points, R, Delta, N, G, num_pt_samples, fix_t_at_t_star):

    method, go_parallel = Laplace_approach, False

    phi_samples = np.zeros([G, num_pt_samples])
    phi_weights = np.zeros(num_pt_samples)
    sample_index = 0

    # Read in t, phi, log_E, and w_sample_mean from MAP curve points
    ts = sp.array([p.t for p in points])
    phis = sp.array([p.phi for p in points])
    log_Es = sp.array([p.log_E for p in points])
    w_sample_means = sp.array([p.sample_mean for p in points])

    # Generate a "histogram" of t according to their relative probability
    num_t = len(ts)
    if fix_t_at_t_star:
        hist_t = np.zeros(num_t)
        hist_t[log_Es.argmax()] = num_pt_samples
    else:
        log_Es = log_Es - log_Es.max()
        prob_t = sp.exp(log_Es)
        prob_t = prob_t / sp.sum(prob_t)
        num_indices = num_t
        sampled_indices = list(
            np.random.choice(num_indices, size=num_pt_samples, replace=True,
                             p=prob_t))
        hist_t = [sampled_indices.count(c) for c in range(num_indices)]

    # Traverse through t, and draw a number of phi samples for each t
    for i in range(num_t):
        num_samples = int(hist_t[i])
        if num_samples > 0:
            t = ts[i]
            phi_t = phis[i]
            phi_samples_at_t, phi_weights_at_t = \
                method(phi_t, R, Delta, t, N, num_samples, go_parallel,
                       pt_sampling=True)
            for k in range(num_samples):
                phi_samples[:, sample_index] = phi_samples_at_t[:, k]

                # JBK: I don't understand this
                phi_weights[sample_index] = phi_weights_at_t[k] / \
                                            w_sample_means[i]

                sample_index += 1

    # Convert phi samples to Q samples
    Q_samples = np.zeros([G, num_pt_samples])
    for k in range(num_pt_samples):
        Q_samples[:, k] = utils.field_to_prob(
            sp.array(phi_samples[:, k]).ravel())

    # Return Q samples along with their weights
    return Q_samples, phi_samples, phi_weights

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

PHI_STD_REG = utils.PHI_STD_REG 

# Compute field from its coefficients in a basis
def coeffs_to_field(coeffs, kernel):
    """ For maxent algorithm. """

    # Get number of gridpoints and dimension of kernel
    G = kernel.shape[0]
    kernel_dim = kernel.shape[1]
    
    # Make sure coeffs is valid
    if not (len(coeffs) == kernel_dim):
        raise ControlledError('/coeffs_to_field/ coeffs must have length %d: len(coeffs) = %d' % (kernel_dim, len(coeffs)))
    if not all(np.isreal(coeffs)):
        raise ControlledError('/coeffs_to_field/ coeffs is not real: coeffs = %s' % coeffs)
    if not all(np.isfinite(coeffs)):
        raise ControlledError('/coeffs_to_field/ coeffs is not finite: coeffs = %s' % coeffs)
    
    # Convert to matrices
    kernel_mat = sp.mat(kernel) # G x kernel_dim matrix
    coeffs_col = sp.mat(coeffs).T # kernel_dim x 1 matrix
    field_col = kernel_mat*coeffs_col # G x 1 matrix

    return sp.array(field_col).ravel() # Returns an array

# Compute the action of a field given its coefficients in a basis
def action_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=False):
    """ For optimizer. Computes action from coefficients. """

    # Get number of gridpoints and dimension of kernel
    G = kernel.shape[0]
    kernel_dim = kernel.shape[1]

    # Make sure coeffs is valid
    if not (len(coeffs) == kernel_dim):
        raise ControlledError('/action_per_datum_from_coeffs/ coeffs must have length %d: len(coeffs) = %d' % (kernel_dim, len(coeffs)))
    if not all(np.isreal(coeffs)):
        raise ControlledError('/action_per_datum_from_coeffs/ coeffs is not real: coeffs = %s' % coeffs)
    if not all(np.isfinite(coeffs)):
        raise ControlledError('/action_per_datum_from_coeffs/ coeffs is not finite: coeffs = %s' % coeffs)
    # Make sure phi0 is valid
    if not isinstance(phi0, np.ndarray):
        phi0 = np.zeros(G)
    else:
        if not all(np.isreal(phi0)):
            raise ControlledError('/action_per_datum_from_coeffs/ phi0 is not real: phi0 = %s' % phi0)
        if not all(np.isfinite(phi0)):
            raise ControlledError('/action_per_datum_from_coeffs/ phi0 is not finite: phi0 = %s' % phi0)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/action_per_datum_from_coeffs/ regularized must be a boolean: regularized = %s' % type(regularized))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    
    current_term = sp.sum(R*phi)
    nonlinear_term = sp.sum(quasiQ)
    s = current_term + nonlinear_term
    
    if regularized:
        s += (.5/G)*sum(phi**2)/(PHI_STD_REG**2)

    # Make sure s is valid
    if not np.isreal(s):
        raise ControlledError('/action_per_datum_from_coeffs/ s is not real: s = %s' % s)
    if not np.isfinite(s):
        raise ControlledError('/action_per_datum_from_coeffs/ s is not finite: s = %s' % s)
        
    return s

# Compute the action gradient w.r.t field coefficients in a basis
def gradient_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=False):
    """ For optimizer. Computes gradient from coefficients. """

    # Get number of gridpoints and dimension of kernel
    G = kernel.shape[0]
    kernel_dim = kernel.shape[1]

    # Make sure coeffs is valid
    if not (len(coeffs) == kernel_dim):
        raise ControlledError('/gradient_per_datum_from_coeffs/ coeffs must have length %d: len(coeffs) = %d' % (kernel_dim, len(coeffs)))
    if not all(np.isreal(coeffs)):
        raise ControlledError('/gradient_per_datum_from_coeffs/ coeffs is not real: coeffs = %s' % coeffs)
    if not all(np.isfinite(coeffs)):
        raise ControlledError('/gradient_per_datum_from_coeffs/ coeffs is not finite: coeffs = %s' % coeffs)
    # Make sure phi0 is valid
    if not isinstance(phi0, np.ndarray):
        phi0 = np.zeros(G)
    else:
        if not all(np.isreal(phi0)):
            raise ControlledError('/gradient_per_datum_from_coeffs/ phi0 is not real: phi0 = %s' % phi0)
        if not all(np.isfinite(phi0)):
            raise ControlledError('/gradient_per_datum_from_coeffs/ phi0 is not finite: phi0 = %s' % phi0)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/gradient_per_datum_from_coeffs/ regularized must be a boolean: regularized = %s' % type(regularized))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    
    R_row = sp.mat(R) # 1 x G
    quasiQ_row = sp.mat(quasiQ) # 1 x G
    kernel_mat = sp.mat(kernel) # G x kernel_dim

    mu_R_row = R_row*kernel_mat # 1 x kernel_dim
    mu_quasiQ_row = quasiQ_row*kernel_mat # 1 x kernel_dim
    grad_row = mu_R_row - mu_quasiQ_row # 1 x kernel_dim
    
    if regularized:
        reg_row = (1./G)*sp.mat(phi)/(PHI_STD_REG**2) # 1 x G
        mu_reg_row = reg_row*kernel_mat # 1 x kernel_dim
        grad_row += mu_reg_row # 1 x kernel_dim

    # Make sure grad_array is valid
    grad_array = sp.array(grad_row).ravel()
    if not all(np.isreal(grad_array)):
        raise ControlledError('/gradient_per_datum_from_coeffs/ grad_array is not real: grad_array = %s' % grad_array)
    if not all(np.isfinite(grad_array)):
        raise ControlledError('/gradient_per_datum_from_coeffs/ grad_array is not finite: grad_array = %s' % grad_array)
        
    return sp.array(grad_row).ravel() # Returns an array

# Compute the action hessian w.r.t field coefficients in a basis
def hessian_per_datum_from_coeffs(coeffs, R, kernel, phi0=False, 
    regularized=False):
    """ For optimizer. Computes hessian from coefficients. """

    # Get number of gridpoints and dimension of kernel
    G = kernel.shape[0]
    kernel_dim = kernel.shape[1]

    # Make sure coeffs is valid
    if not (len(coeffs) == kernel_dim):
        raise ControlledError('/hessian_per_datum_from_coeffs/ coeffs must have length %d: len(coeffs) = %d' % (kernel_dim, len(coeffs)))
    if not all(np.isreal(coeffs)):
        raise ControlledError('/hessian_per_datum_from_coeffs/ coeffs is not real: coeffs = %s' % coeffs)
    if not all(np.isfinite(coeffs)):
        raise ControlledError('/hessian_per_datum_from_coeffs/ coeffs is not finite: coeffs = %s' % coeffs)
    # Make sure phi0 is valid
    if not isinstance(phi0, np.ndarray):
        phi0 = np.zeros(G)
    else:
        if not all(np.isreal(phi0)):
            raise ControlledError('/hessian_per_datum_from_coeffs/ phi0 is not real: phi0 = %s' % phi0)
        if not all(np.isfinite(phi0)):
            raise ControlledError('/hessian_per_datum_from_coeffs/ phi0 is not finite: phi0 = %s' % phi0)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/hessian_per_datum_from_coeffs/ regularized must be a boolean: regularized = %s' % type(regularized))

    phi = coeffs_to_field(coeffs, kernel)
    quasiQ = utils.field_to_quasiprob(phi+phi0)
    
    kernel_mat = sp.mat(kernel) # G x kernel_dim 
    H = sp.mat(sp.diag(quasiQ)) # G x G
    
    if regularized:
        H += (1./G)*sp.diag(np.ones(G))/(PHI_STD_REG**2)
        
    hessian_mat = kernel_mat.T*H*kernel_mat # kernel_dim x kernel_dim

    # Make sure hessian_array is valid ?

    return sp.array(hessian_mat) # Returns an array

# Computes the maximum entropy probaiblity distribution
def compute_maxent_prob_1d(R, kernel, h=1.0, report_num_steps=False, 
    phi0=False):
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    field, num_corrector_steps, num_backtracks = \
        compute_maxent_field(R, kernel, report_num_steps=True, phi0=phi0)
    Q = utils.field_to_prob(field+phi0)/h
    if report_num_steps:
        return Q, num_corrector_steps, num_backtracks
    else:
        return Q

# Computes the maximum entropy probaiblity distribution
def compute_maxent_prob_2d(R, kernel, grid_spacing=[1.0,1.0],\
        report_num_steps=False, phi0=False):
    if not isinstance(phi0,np.ndarray):
        phi0 = np.zeros(R.size)
    else:
        assert all(np.isreal(phi0))

    phi, num_corrector_steps, num_backtracks = \
        compute_maxent_field(R, kernel, report_num_steps=True)
    h = grid_spacing[0]*grid_spacing[1]
    Q = utils.field_to_prob(phi+phi0)/h
    if report_num_steps:
        return Q, num_corrector_steps, num_backtracks
    else:
        return Q

# Compute the maxent field 
def compute_maxent_field(R, kernel, report_num_steps=False, 
    phi0=False, geo_dist_tollerance=1E-3, grad_tollerance=1E-5):
    """
    Computes the maxent field from a histogram and kernel
    
    Args:
        R (numpy.narray): 
            Normalized histogram of the raw data. Should have size G
            
        kernel (numpy.ndarray): 
            Array of vectors spanning the smoothness operator kernel. Should
            have size G x kernel_dim
            
    Returns:
        
        phi: 
            The MaxEnt field. 
    """
    
    # Make sure report_num_steps is valid
    if not isinstance(report_num_steps, bool):
        raise ControlledError('/compute_maxent_field/ report_num_steps must be a boolean: report_num_steps = %s' % type(report_num_steps))
    # Make sure phi0 is valid
    if not isinstance(phi0, np.ndarray):
        phi0 = np.zeros(len(R))
    else:
        if not all(np.isreal(phi0)):
            raise ControlledError('/compute_maxent_field/ phi0 is not real: phi0 = %s' % phi0)
        if not all(np.isfinite(phi0)):
            raise ControlledError('/compute_maxent_field/ phi0 is not finite: phi0 = %s' % phi0)
    # Make sure geo_dist_tollerance is valid
    if not isinstance(geo_dist_tollerance, float):
        raise ControlledError('/compute_maxent_field/ geo_dist_tollerance must be a float: geo_dist_tollerance = %s' % type(geo_dist_tollerance))
    # Make sure grad_tollerance is valid
    if not isinstance(grad_tollerance, float):
        raise ControlledError('/compute_maxent_field/ grad_tollerance must be a float: grad_tollerance = %s' % type(grad_tollerance))
        
    # Get number of gridpoints and dimension of kernel
    G = kernel.shape[0]
    kernel_dim = kernel.shape[1]

    # Set coefficients to zero
    if kernel_dim > 1:
        coeffs = sp.zeros(kernel_dim)
        #coeffs = sp.randn(kernel_dim)
    else:
        coeffs = sp.zeros(1)

    # Evaluate the probabiltiy distribution
    phi = coeffs_to_field(coeffs, kernel)
    phi = sp.array(phi).ravel()
    phi0 = sp.array(phi0).ravel()
    #print phi+phi0
    Q = utils.field_to_prob(phi+phi0)

    # Evaluate action
    s = action_per_datum_from_coeffs(coeffs, R, kernel, phi0)

    # Perform corrector steps until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:
        
        if kernel_dim == 1:
            success = True
            break
        
        # Compute the gradient
        v = gradient_per_datum_from_coeffs(coeffs, R, kernel, phi0)
        
        # If gradient is not detectable, we're already done!
        if norm(v) < G*utils.TINY_FLOAT32:
            break

        # Compute the hessian
        Lambda = hessian_per_datum_from_coeffs(coeffs, R, kernel, phi0) 

        # Solve linear equation to get change in field
        # This is the conjugate gradient method
        da = -sp.real(solve(Lambda,v))

        # Compute corresponding change in action
        ds = sp.sum(da*v)

        # This should always be satisifed
        if (ds > 0):
            print('Warning: ds > 0. Quitting compute_maxent_field.')
            break

        # Reduce step size until in linear regime
        beta = 1.0
        success = False
        while True:

            # Compute new phi and new action
            coeffs_new = coeffs + beta*da
            s_new = action_per_datum_from_coeffs(coeffs_new,R,kernel,phi0) 

            # Check for linear regime
            if s_new <= s + 0.5*beta*ds:
                break

            # Check to see if beta is too small and algorithm is failing
            elif beta < 1E-20:
                raise ControlledError('/compute_maxent_field/ phi is not converging: beta = %s' % beta)

            # If not in linear regime backtrack value of beta
            else:
                # pdb.set_trace()
                num_backtracks+=1
                beta *= 0.5

        # Compute new distribution
        phi_new = coeffs_to_field(coeffs_new, kernel) 
        Q_new = utils.field_to_prob(phi_new+phi0) 

        # Break out of loop if Q_new is close enough to Q
        if (utils.geo_dist(Q_new,Q) < geo_dist_tollerance) and (np.linalg.norm(v) < grad_tollerance):
            success = True
            break
        
        # Break out of loop with warning if S_new > S. Should not happen,
        # but not fatal if it does. Just means less precision
        elif s_new-s > 0:
            print('Warning: action has increased. Terminating steps.')
            success = False
            break

        # Otherwise, continue with corrector step
        else:
            num_corrector_steps += 1

            # Set new coefficients.
            # New s, Q, and phi laready computed
            coeffs = coeffs_new
            s = s_new
            Q = Q_new
            phi = phi_new

    # Actually, should judge success by whether moments match
    if not success:
        print('gradident norm == %f'%np.linalg.norm(v))
        print('gradient tollerance == %f'%grad_tollerance)
        print('Failure! Trying Maxent again!')
        
    # After corrector loop has finished, return field
    # Also return stepping stats if requested
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi, success

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Put hard bounds on how big or small t can be. T_MIN especially seems to help convergence
T_MAX = 40
T_MIN = -40
PHI_MAX = utils.PHI_MAX
PHI_MIN = utils.PHI_MIN
MAX_DS = -1E-3
PHI_STD_REG = utils.PHI_STD_REG


class Results():
    pass


# Represents a point along the MAP curve
class MAP_curve_point:
    def __init__(self, t, phi, Q, log_E, sample_mean, sample_mean_std_dev, details=False):
        self.t = t
        self.phi = phi
        self.Q = Q
        self.log_E = log_E
        self.sample_mean = sample_mean
        self.sample_mean_std_dev = sample_mean_std_dev
        # self.details = details


# Represents the MAP curve
class MAP_curve:
    def __init__(self):
        self.points = []
        self._is_sorted = False

    def add_point(self, t, phi, Q, log_E, sample_mean, sample_mean_std_dev, details=False):
        point = MAP_curve_point(t, phi, Q, log_E, sample_mean, sample_mean_std_dev, details)
        self.points.append(point)
        self._is_sorted = False

    def sort(self):
        self.points.sort(key=lambda x: x.t)
        self._is_sorted = True

    # Use this to get actual points along the MAP curve. This ensures that points are sorted
    def get_points(self):
        if not self._is_sorted:
            self.sort()
        return self.points

    def get_maxent_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[0]
        if not (p.t == -sp.Inf):
            raise ControlledError('/MAP_curve/ Not getting MaxEnt point: t = %f' % p.t)
        return p

    def get_histogram_point(self):
        if not self._is_sorted:
            self.sort()
        p = self.points[-1]
        if not (p.t == sp.Inf):
            raise ControlledError('/MAP_curve/ Not getting histogram point: t = %f' % p.t)
        return p

    def get_log_evidence_ratios(self, finite=True):
        log_Es = sp.array([p.log_E for p in self.points])
        ts = sp.array([p.t for p in self.points])
        if finite:
            indices = (log_Es > -np.Inf) * (ts > -np.Inf) * (ts < np.Inf)
            return log_Es[indices], ts[indices]
        else:
            return log_Es, ts


#
# Convention: action, gradient, and hessian are G/N * the actual. This provides for more robust numerics
#
# Evaluate the action of a field given smoothness criteria
def action(phi, R, Delta, t, N, phi_in_kernel=False, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/action/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/action/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/action/ t is not real: t = %s' % t)
    # if not np.isfinite(t):
    #    raise ControlledError('/action/ t is not finite: t = %s' % t)
    # Make sure phi_in_kernel is valid
    if not isinstance(phi_in_kernel, bool):
        raise ControlledError('/action/ phi_in_kernel must be a boolean: phi_in_kernel = %s' % type(phi_in_kernel))
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/action/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    ones_col = sp.mat(sp.ones(int(G))).T

    if phi_in_kernel:
        S_mat = G * R_col.T * phi_col + G * ones_col.T * quasiQ_col
    else:
        S_mat = 0.5 * sp.exp(
            -t) * phi_col.T * Delta_sparse * phi_col + G * R_col.T * phi_col + G * ones_col.T * quasiQ_col

    if regularized:
        S_mat += 0.5 * (phi_col.T * phi_col) / (N * PHI_STD_REG ** 2)

    S = S_mat[0, 0]

    # Make sure S is valid
    if not np.isreal(S):
        raise ControlledError('/action/ S is not real at t = %s: S = %s' % (t, S))
    if not np.isfinite(S):
        raise ControlledError('/action/ S is not finite at t = %s: S = %s' % (t, S))

    return S


# Evaluate action gradient w.r.t. a field given smoothness criteria
def gradient(phi, R, Delta, t, N, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/gradient/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/gradient/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/gradient/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/gradient/ t is not finite: t = %s' % t)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/gradient/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    quasiQ_col = sp.mat(quasiQ).T
    Delta_sparse = Delta.get_sparse_matrix()
    phi_col = sp.mat(phi).T
    R_col = sp.mat(R).T
    grad_col = sp.exp(-t) * Delta_sparse * phi_col + G * R_col - G * quasiQ_col

    if regularized:
        grad_col += phi_col / (N * PHI_STD_REG ** 2)

    grad = sp.array(grad_col).ravel()

    # Make sure grad is valid
    if not all(np.isreal(grad)):
        raise ControlledError('/gradient/ grad is not real at t = %s: grad = %s' % (t, grad))
    if not all(np.isfinite(grad)):
        raise ControlledError('/gradient/ grad is not finite at t = %s: grad = %s' % (t, grad))

    return grad


# Evaluate action hessian w.r.t. a field given smoothness criteria. NOTE: returns sparse matrix, not dense matrix!
def hessian(phi, R, Delta, t, N, regularized=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/hessian/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/hessian/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/hessian/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/hessian/ t is not finite: t = %s' % t)
    # Make sure regularized is valid
    if not isinstance(regularized, bool):
        raise ControlledError('/hessian/ regularized must be a boolean: regularized = %s' % type(regularized))

    G = 1. * len(R)
    quasiQ = utils.field_to_quasiprob(phi)
    Delta_sparse = Delta.get_sparse_matrix()
    H = sp.exp(-t) * Delta_sparse + G * diags(quasiQ, 0)

    if regularized:
        H += diags(np.ones(int(G)), 0) / (N * PHI_STD_REG ** 2)

    # Make sure H is valid ?
    return H


# Compute the log of ptgd at maxent
def log_ptgd_at_maxent(phi_M, R, Delta, N, Z_eval, num_Z_samples):
    # Make sure phi_M is valid
    if not all(np.isreal(phi_M)):
        raise ControlledError('/log_ptgd_at_maxent/ phi_M is not real: phi_M = %s' % phi_M)
    if not all(np.isfinite(phi_M)):
        raise ControlledError('/log_ptgd_at_maxent/ phi_M is not finite: phi_M = %s' % phi_M)

    kernel_dim = Delta._kernel_dim
    M = utils.field_to_prob(phi_M)
    M_on_kernel = sp.zeros([kernel_dim, kernel_dim])
    kernel_basis = Delta._kernel_basis
    lambdas = Delta._eigenvalues
    for a in range(int(kernel_dim)):
        for b in range(int(kernel_dim)):
            psi_a = sp.ravel(kernel_basis[:, a])
            psi_b = sp.ravel(kernel_basis[:, b])
            M_on_kernel[a, b] = sp.sum(psi_a * psi_b * M)

    # Compute log Occam factor at infinity
    log_Occam_at_infty = - 0.5 * sp.log(det(M_on_kernel)) - 0.5 * sp.sum(sp.log(lambdas[kernel_dim:]))

    # Make sure log_Occam_at_infty is valid
    if not np.isreal(log_Occam_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_Occam_at_infty is not real: log_Occam_at_infty = %s' %
                              log_Occam_at_infty)
    if not np.isfinite(log_Occam_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_Occam_at_infty is not finite: log_Occam_at_infty = %s' %
                              log_Occam_at_infty)

    # Compute the log likelihood at infinity
    log_likelihood_at_infty = - N * sp.sum(phi_M * R) - N

    # Make sure log_likelihood_at_infty is valid
    if not np.isreal(log_likelihood_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_likelihood_at_infty is not real: log_likelihood_at_infty = %s' %
                              log_likelihood_at_infty)
    if not np.isfinite(log_likelihood_at_infty):
        raise ControlledError('/log_ptgd_at_maxent/ log_likelihood_at_infty is not finite: log_likelihood_at_infty = %s' %
                              log_likelihood_at_infty)

    # Compute the log posterior (not sure this is right)
    log_ptgd_at_maxent = log_likelihood_at_infty + log_Occam_at_infty

    # If requested, incorporate corrections to the partition function
    t = -np.inf
    num_samples = num_Z_samples
    if Z_eval == 'Lap':
        correction, w_sample_mean, w_sample_mean_std = \
            0.0, 1.0, 0.0
    if Z_eval == 'Lap+Imp':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False)
    if Z_eval == 'Lap+Imp+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True)
    if Z_eval == 'GLap':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False, sampling=False)
    if Z_eval == 'GLap+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True, sampling=False)
    if Z_eval == 'GLap+Sam':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=False, sampling=True)
    if Z_eval == 'GLap+Sam+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi_M, R, Delta, t, N, num_samples, go_parallel=True, sampling=True)
    if Z_eval == 'Lap+Fey':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Feynman_diagrams(phi_M, R, Delta, t, N)

    # Make sure correction is valid
    if not np.isreal(correction):
        raise ControlledError('/log_ptgd_at_maxent/ correction is not real: correction = %s' % correction)
    if not np.isfinite(correction):
        raise ControlledError('/log_ptgd_at_maxent/ correction is not finite: correction = %s' % correction)

    log_ptgd_at_maxent += correction

    return log_ptgd_at_maxent, w_sample_mean, w_sample_mean_std


# Computes the log of ptgd at t
def log_ptgd(phi, R, Delta, t, N, Z_eval, num_Z_samples):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/log_ptgd/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/log_ptgd/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/log_ptgd/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/log_ptgd/ t is not finite: t = %s' % t)

    G = 1. * len(phi)
    alpha = 1. * Delta._alpha
    kernel_dim = 1. * Delta._kernel_dim
    H = hessian(phi, R, Delta, t, N)
    H_prime = H.todense() * sp.exp(t)

    S = action(phi, R, Delta, t, N)

    # First try computing log determinant straight away
    log_det = sp.log(det(H_prime))

    # If failed, try computing the sum of eigenvalues, forcing the eigenvalues to be real and non-negative
    if not (np.isreal(log_det) and np.isfinite(log_det)):
        lambdas = abs(eigvalsh(H_prime))
        log_det = sp.sum(sp.log(lambdas))

        # Make sure log_det is valid
    if not np.isreal(log_det):
        raise ControlledError('/log_ptgd/ log_det is not real at t = %s: log_det = %s' % (t, log_det))
    if not np.isfinite(log_det):
        raise ControlledError('/log_ptgd/ log_det is not finite at t = %s: log_det = %s' % (t, log_det))

    # Compute contribution from finite t
    log_ptgd = -(N / G) * S + 0.5 * kernel_dim * t - 0.5 * log_det

    # Make sure log_ptgd is valid
    if not np.isreal(log_ptgd):
        raise ControlledError('/log_ptgd/ log_ptgd is not real at t = %s: log_ptgd = %s' % (t, log_ptgd))
    if not np.isfinite(log_ptgd):
        raise ControlledError('/log_ptgd/ log_ptgd is not finite at t = %s: log_ptgd = %s' % (t, log_ptgd))

    # If requested, incorporate corrections to the partition function
    num_samples = num_Z_samples
    if Z_eval == 'Lap':
        correction, w_sample_mean, w_sample_mean_std = \
            0.0, 1.0, 0.0
    if Z_eval == 'Lap+Imp':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False)
    if Z_eval == 'Lap+Imp+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Laplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True)
    if Z_eval == 'GLap':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False, sampling=False)
    if Z_eval == 'GLap+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True, sampling=False)
    if Z_eval == 'GLap+Sam':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=False, sampling=True)
    if Z_eval == 'GLap+Sam+P':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.GLaplace_approach(phi, R, Delta, t, N, num_samples, go_parallel=True, sampling=True)
    if Z_eval == 'Lap+Fey':
        correction, w_sample_mean, w_sample_mean_std = \
            supplements.Feynman_diagrams(phi, R, Delta, t, N)

    # Make sure correction is valid
    if not np.isreal(correction):
        raise ControlledError('/log_ptgd/ correction is not real at t = %s: correction = %s' % (t, correction))
    if not np.isfinite(correction):
        raise ControlledError('/log_ptgd/ correction is not finite at t = %s: correction = %s' % (t, correction))

    log_ptgd += correction

    details = Results()
    details.S = S
    details.N = N
    details.G = G
    details.kernel_dim = kernel_dim
    details.t = t
    details.log_det = log_det
    details.phi = phi

    return log_ptgd, w_sample_mean, w_sample_mean_std


# Computes predictor step
def compute_predictor_step(phi, R, Delta, t, N, direction, resolution, DT_MAX):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/compute_predictor_step/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/compute_predictor_step/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/compute_predictor_step/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/compute_predictor_step/ t is not finite: t = %s' % t)
    # Make sure direction is valid
    if not ((direction == 1) or (direction == -1)):
        raise ControlledError('/compute_predictor_step/ direction must be just a sign: direction = %s' % direction)

    # Get current probability distribution
    Q = utils.field_to_prob(phi)
    G = 1. * len(Q)

    # Get hessian
    H = hessian(phi, R, Delta, t, N)

    # Compute rho, which indicates direction of step
    rho = G * spsolve(H, Q - R)

    # Make sure rho is valid
    if not all(np.isreal(rho)):
        raise ControlledError('/compute_predictor_step/ rho is not real at t = %s: rho = %s' % (t, rho))
    if not all(np.isfinite(rho)):
        raise ControlledError('/compute_predictor_step/ rho is not finite at t = %s: rho = %s' % (t, rho))

    denom = sp.sqrt(sp.sum(rho * Q * rho))

    # Make sure denom is valid
    if not np.isreal(denom):
        raise ControlledError('/compute_predictor_step/ denom is not real at t = %s: denom = %s' % (t, denom))
    if not np.isfinite(denom):
        raise ControlledError('/compute_predictor_step/ denom is not finite at t = %s: denom = %s' % (t, denom))
    if not (denom > 0):
        raise ControlledError('/compute_predictor_step/ denom is not positive at t = %s: denom = %s' % (t, denom))

    # Compute dt based on value of epsilon (the resolution)
    dt = direction * resolution / denom
    while abs(dt) > DT_MAX:
        dt /= 2.0

        # Return phi_new and new t_new. WARNING: IT IS NOT YET CLEAR THAT PHI_NEW ISN'T INSANE
    phi_new = phi + rho * dt
    t_new = t + dt

    # Make sure phi_new is valid
    if not all(np.isreal(phi_new)):
        raise ControlledError('/compute_predictor_step/ phi_new is not real at t_new = %s: phi_new = %s' % (t_new, phi_new))
    if not all(np.isfinite(phi_new)):
        raise ControlledError('/compute_predictor_step/ phi_new is not finite at t_new = %s: phi_new = %s' % (t_new, phi_new))
    # Make sure t_new is valid
    if not np.isreal(t_new):
        raise ControlledError('/compute_predictor_step/ t_new is not real: t_new = %s' % t_new)
    if not np.isfinite(t_new):
        raise ControlledError('/compute_predictor_step/ t_new is not finite: t_new = %s' % t_new)

    return phi_new, t_new


# Computes corrector step
def compute_corrector_step(phi, R, Delta, t, N, tollerance, report_num_steps=False):
    # Make sure phi is valid
    if not all(np.isreal(phi)):
        raise ControlledError('/compute_corrector_step/ phi is not real: phi = %s' % phi)
    if not all(np.isfinite(phi)):
        raise ControlledError('/compute_corrector_step/ phi is not finite: phi = %s' % phi)
    # Make sure t is valid
    if not np.isreal(t):
        raise ControlledError('/compute_corrector_step/ t is not real: t = %s' % t)
    if not np.isfinite(t):
        raise ControlledError('/compute_corrector_step/ t is not finite: t = %s' % t)
    # Make sure report_num_steps is valid
    if not isinstance(report_num_steps, bool):
        raise ControlledError('/compute_corrector_step/ report_num_steps must be a boolean: report_num_steps = %s' %
                              type(report_num_steps))

    # Evaluate the probability distribution
    Q = utils.field_to_prob(phi)

    # Evaluate action
    S = action(phi, R, Delta, t, N)

    # Perform corrector steps until phi converges
    num_corrector_steps = 0
    num_backtracks = 0
    while True:

        # Compute the gradient
        v = gradient(phi, R, Delta, t, N)

        # Compute the hessian
        H = hessian(phi, R, Delta, t, N)

        # Solve linear equation to get change in field
        dphi = -spsolve(H, v)

        # Make sure dphi is valid
        if not all(np.isreal(dphi)):
            raise ControlledError('/compute_corrector_step/ dphi is not real at t = %s: dphi = %s' % (t, dphi))
        if not all(np.isfinite(dphi)):
            raise ControlledError('/compute_corrector_step/ dphi is not finite at t = %s: dphi = %s' % (t, dphi))

        # Compute corresponding change in action
        dS = sp.sum(dphi * v)

        # If we're already very close to the max, then dS will be close to zero. In this case, we're done already
        if dS > MAX_DS:
            break

        # Reduce step size until in linear regime
        beta = 1.0
        while True:

            # Make sure beta is valid
            if beta < 1E-50:
                raise ControlledError('/compute_corrector_step/ phi is not converging at t = %s: beta = %s' % (t, beta))

            # Compute new phi
            phi_new = phi + beta * dphi

            # If new phi is insane, decrease beta
            if any(phi_new < PHI_MIN) or any(phi_new > PHI_MAX):
                num_backtracks += 1
                beta *= 0.5
                continue

            # Compute new action
            S_new = action(phi_new, R, Delta, t, N)

            # Check for linear regime
            if S_new - S <= 0.5 * beta * dS:
                break

            # If not in linear regime, backtrack value of beta
            else:
                num_backtracks += 1
                beta *= 0.5
                continue

        # Make sure phi_new is valid
        if not all(np.isreal(phi_new)):
            raise ControlledError('/compute_corrector_step/ phi_new is not real at t = %s: phi_new = %s' % (t, phi_new))
        if not all(np.isfinite(phi_new)):
            raise ControlledError('/compute_corrector_step/ phi_new is not finite at t = %s: phi_new = %s' % (t, phi_new))

        # Compute new Q
        Q_new = utils.field_to_prob(phi_new)

        # Break out of loop if Q_new is close enough to Q
        gd = utils.geo_dist(Q_new, Q)
        if gd < tollerance:
            break

        # Break out of loop with warning if S_new > S.
        # Should not happen, but not fatal if it does. Just means less precision
        # ACTUALLY, THIS SHOULD NEVER HAPPEN!
        elif S_new - S > 0:
            raise ControlledError('/compute_corrector_step/ S_new > S at t = %s: terminating corrector steps' % t)

        # Otherwise, continue with corrector step
        else:
            # New phi, Q, and S values have already been computed
            phi = phi_new
            Q = Q_new
            S = S_new
            num_corrector_steps += 1

    # After corrector loop has finished, return field
    if report_num_steps:
        return phi, num_corrector_steps, num_backtracks
    else:
        return phi


# The core algorithm of DEFT, used for both 1D and 2D density estimation
def compute_map_curve(N, R, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t, tollerance, resolution, max_log_evidence_ratio_drop):
    """ Traces the map curve in both directions

    Args:

        R (numpy.narray):
            The data histogram

        Delta (Smoothness_operator instance):
            Effectiely defines smoothness

        resolution (float):
            Specifies max distance between neighboring points on the
            MAP curve

    Returns:

        map_curve (list): A list of MAP_curve_points

    """

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    alpha = Delta._alpha
    kernel_basis = Delta.get_kernel_basis()
    kernel_dim = Delta.get_kernel_dim()

    # Initialize MAP curve
    map_curve = MAP_curve()

    #
    # First compute histogram stuff
    #

    # Get normalized histogram and corresponding field
    R = R / sum(R)
    phi_R = utils.prob_to_field(R)
    log_E_R = -np.Inf
    t_R = np.Inf
    w_sample_mean_R = 1.0
    w_sample_mean_std_R = 0.0
    map_curve.add_point(t_R, phi_R, R, log_E_R, w_sample_mean_R, w_sample_mean_std_R)

    #
    # Then compute maxent stuff
    #

    # Compute the maxent field and density
    phi_infty, success = maxent.compute_maxent_field(R, kernel_basis)

    # Convert maxent field to probability distribution
    M = utils.field_to_prob(phi_infty)

    # Compute the maxent log_ptgd. Important to keep this around to compute log_E at finite t
    log_ptgd_M, w_sample_mean_M, w_sample_mean_std_M = \
        log_ptgd_at_maxent(phi_infty, R, Delta, N, Z_eval, num_Z_samples)

    # This corresponds to a log_E of zero
    log_E_M = 0
    t_M = -sp.Inf
    map_curve.add_point(t_M, phi_infty, M, log_E_M, w_sample_mean_M, w_sample_mean_std_M)

    # Set maximum log evidence ratio so far encountered
    log_E_max = -np.Inf

    #
    # Now compute starting point
    #

    # Compute phi_start by executing a corrector step starting at maxent dist
    phi_start = compute_corrector_step(phi_infty, R, Delta, t_start, N, tollerance)

    # Convert starting field to probability distribution
    Q_start = utils.field_to_prob(phi_start)

    # Compute log ptgd
    log_ptgd_start, w_sample_mean_start, w_sample_mean_std_start = \
        log_ptgd(phi_start, R, Delta, t_start, N, Z_eval, num_Z_samples)

    # Compute corresponding evidence ratio
    log_E_start = log_ptgd_start - log_ptgd_M

    # Adjust max log evidence ratio
    log_E_max = log_E_start if (log_E_start > log_E_max) else log_E_max

    # Set start as first MAP curve point
    if print_t:
        print('t = %.2f' % t_start)
    map_curve.add_point(t_start, phi_start, Q_start, log_E_start, w_sample_mean_start, w_sample_mean_std_start)

    #
    # Finally trace along the MAP curve
    #

    # This is to indicate how iteration in t is terminated
    break_t_loop = [True, True]  # = [Q_M, Q_R]; True = thru geo_dist, False = thru log_E

    # Trace MAP curve in both directions
    for direction in [-1, +1]:

        # Start iteration from central point
        phi = phi_start
        t = t_start
        Q = Q_start
        log_E = log_E_start
        w_sample_mean = w_sample_mean_start
        w_sample_mean_std_dev = w_sample_mean_std_start

        if direction == -1:
            Q_end = M
        else:
            Q_end = R

        log_ptgd0 = log_ptgd_start
        slope = np.sign(0)

        # Keep stepping in direction until reach the specified endpoint
        while True:

            # Test distance to endpoint
            if utils.geo_dist(Q_end, Q) <= resolution:
                if direction == -1:
                    pass
                    #print('Q_end = M: geo_dist (%.2f) <= resolution (%.2f)' % (utils.geo_dist(Q_end, Q), resolution))
                else:
                    pass
                    #print('Q_end = R: geo_dist (%.2f) <= resolution (%.2f)' % (utils.geo_dist(Q_end, Q), resolution))
                break

            # Take predictor step
            phi_pre, t_new = compute_predictor_step(phi, R, Delta, t, N, direction, resolution, DT_MAX)

            # If phi_pre is insane, start iterating from phi instead
            if any(phi_pre > PHI_MAX) or any(phi_pre < PHI_MIN):
                phi_pre = phi

            # Perform corrector steps to get new phi
            phi_new = compute_corrector_step(phi_pre, R, Delta, t_new, N, tollerance)

            # Compute new distribution
            Q_new = utils.field_to_prob(phi_new)

            # Compute log ptgd
            log_ptgd_new, w_sample_mean_new, w_sample_mean_std_new = \
                log_ptgd(phi_new, R, Delta, t_new, N, Z_eval, num_Z_samples)

            # Compute corresponding evidence ratio
            log_E_new = log_ptgd_new - log_ptgd_M

            # Take step
            t = t_new
            Q = Q_new
            phi = phi_new
            log_E = log_E_new
            w_sample_mean = w_sample_mean_new
            w_sample_mean_std = w_sample_mean_std_new

            # Adjust max log evidence ratio
            log_E_max = log_E if (log_E > log_E_max) else log_E_max

            # Terminate if log_E is too small. But don't count the t=-inf endpoint when computing log_E_max
            if log_E_new < log_E_max - max_log_evidence_ratio_drop:
                if direction == -1:
                    #print('Q_end = M: log_E (%.2f) < log_E_max (%.2f) - max_log_evidence_ratio_drop (%.2f)' %
                    #      (log_E_new, log_E_max, max_log_evidence_ratio_drop))
                    break_t_loop[0] = False
                else:
                    #print('Q_end = R: log_E (%.2f) < log_E_max (%.2f) - max_log_evidence_ratio_drop (%.2f)' %
                    #      (log_E_new, log_E_max, max_log_evidence_ratio_drop))
                    break_t_loop[1] = False
                # Add new point to map curve
                if print_t:
                    print('t = %.2f' % t)
                map_curve.add_point(t, phi, Q, log_E, w_sample_mean, w_sample_mean_std)
                break

            slope_new = np.sign(log_ptgd_new - log_ptgd0)
            # Terminate if t is too negative or too positive
            if t < T_MIN:
                #print('Q_end = M: t (%.2f) < T_MIN (%.2f)' % (t, T_MIN))
                break_t_loop[0] = False
                break
            elif t > T_MAX:
                #print('Q_end = R: t (%.2f) > T_MAX (%.2f)' % (t, T_MAX))
                break_t_loop[1] = False
                break
            elif (direction == +1) and (t > 0) and (np.sign(slope_new * slope) < 0) and (log_ptgd_new > log_ptgd0):
                #print('Q_end = R: t (%.2f) > 0 and log_ptgd_new (%.2f) > log_ptgd (%.2f) wrongly' %
                #      (t, log_ptgd_new, log_ptgd0))
                break_t_loop[1] = False
                break
            elif (direction == +1) and (np.sign(slope_new * slope) < 0) and (log_ptgd_new > log_ptgd0 + max_log_evidence_ratio_drop):
                #print('Q_end = R: log_ptgd_new (%.2f) > log_ptgd (%.2f) + max_log_evidence_ratio_drop (%.2f) at t = %.2f' %
                #      (log_ptgd_new, log_ptgd0, max_log_evidence_ratio_drop, t))
                break_t_loop[1] = False
                break
            log_ptgd0 = log_ptgd_new
            slope = slope_new

            # Add new point to MAP curve
            if print_t:
                print('t = %.2f' % t)
            map_curve.add_point(t, phi, Q, log_E, w_sample_mean, w_sample_mean_std)

    # Sort points along the MAP curve
    map_curve.sort()
    map_curve.t_start = t_start
    map_curve.break_t_loop = break_t_loop

    # Return the MAP curve to the user
    return map_curve


#
# Core DEFT algorithm
#
def run(counts_array, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t,
        tollerance, resolution, num_pt_samples, fix_t_at_t_star,max_log_evidence_ratio_drop, details=False):
    """
    The core algorithm of DEFT, used for both 1D and 2D density estmation.

    Args:
        counts_array (numpy.ndarray):
            A scipy array of counts. All counts must be nonnegative.

        Delta (Smoothness_operator instance):
            An operator providing the definition of 'smoothness' used by DEFT
    """

    # Make sure details is valid
    if not isinstance(details, bool):
        raise ControlledError('/deft_core._run/ details must be a boolean: details = %s' % type(details))

    # Get number of gridpoints and kernel dimension from smoothness operator
    G = Delta.get_G()
    kernel_dim = Delta.get_kernel_dim()

    # Make sure counts_array is valid
    if not (len(counts_array) == G):
        raise ControlledError('/deft_core._run/ counts_array must have length %d: len(counts_array) = %d' %
                              (G, len(counts_array)))
    if not all(counts_array >= 0):
        raise ControlledError('/deft_core._run/ counts_array is not non-negative: counts_array = %s' % counts_array)
    if not (sum(counts_array > 0) > kernel_dim):
        raise ControlledError('/deft_core._run/ Only %d elements of counts_array contain data, less than kernel dimension %d' %
                              (sum(counts_array > 0), kernel_dim))

    # Get number of data points and normalize histogram
    N = sum(counts_array)
    R = 1.0 * counts_array / N

    #
    # Compute the MAP curve
    #

    clock = ConsumedTimeTimer()
    clock.tic()
    map_curve = compute_map_curve(N, R, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t, tollerance, resolution,max_log_evidence_ratio_drop)
    map_curve_compute_time = clock.toc()
    if print_t:
        print('MAP curve computation took %.2f sec' % map_curve_compute_time)

    # Identify the optimal density estimate
    points = map_curve.points
    log_Es = sp.array([p.log_E for p in points])
    log_E_max = log_Es.max()
    ibest = log_Es.argmax()
    star = points[ibest]
    Q_star = np.copy(star.Q)
    t_star = star.t
    phi_star = np.copy(star.phi)
    map_curve.i_star = ibest

    #
    # Do posterior sampling
    #

    if not (num_pt_samples == 0):
        Q_samples, phi_samples, phi_weights = \
            supplements.posterior_sampling(points, R, Delta, N, G,
                                           num_pt_samples, fix_t_at_t_star)



    #
    # Package results
    #

    # Create a container
    results = Results()

    # Fill in info that's guaranteed to be there
    results.phi_star = phi_star
    results.Q_star = Q_star
    results.R = R
    results.map_curve = map_curve
    results.map_curve_compute_time = map_curve_compute_time
    results.G = G
    results.N = N
    results.t_star = t_star
    results.i_star = ibest
    results.counts = counts_array
    results.tollerance = tollerance
    results.resolution = resolution
    results.points = points

    # Get maxent point
    maxent_point = results.map_curve.get_maxent_point()
    results.M = maxent_point.Q / np.sum(maxent_point.Q)

    # Include posterior sampling info if any sampling was performed
    if not (num_pt_samples == 0):
        results.num_pt_samples = num_pt_samples
        results.Q_samples = Q_samples
        results.phi_samples = phi_samples
        results.phi_weights = phi_weights

    # Return density estimate along with histogram on which it is based
    return results

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

SMALL_NUM = 1E-6
MAX_NUM_GRID_POINTS = 1000
DEFAULT_NUM_GRID_POINTS = 100
MAX_NUM_POSTERIOR_SAMPLES = 1000
MAX_NUM_SAMPLES_FOR_Z = 1000

class DensityEstimator:
    """Estimates a 1D probability density from sampled data.

    parameters
    ----------
    data: (set, list, or np.array of numbers)
        An array of data from which the probability density will be estimated.
        Infinite or NaN values will be discarded.

    grid: (1D np.array)
        An array of evenly spaced grid points on which the probability density
        will be estimated. Default value is ``None``, in which case the grid is
        set automatically.

    grid_spacing: (float > 0)
        The distance at which to space neighboring grid points. Default value
        is ``None``, in which case this spacing is set automatically.

    num_grid_points: (int)
        The number of grid points to draw within the data domain. Restricted
        to ``2*alpha <= num_grid_points <= 1000``. Default value is ``None``, in
        which case the number of grid points is chosen automatically.

    bounding_box: ([float, float])
        The boundaries of the data domain, within which the probability density
        will be estimated. Default value is ``None``, in which case the
        bounding box is set automatically to encompass all of the data.

    alpha: (int)
        The order of derivative constrained in the definition of smoothness.
        Restricted to ``1 <= alpha <= 4``. Default value is 3.

    periodic: (bool)
        Whether or not to impose periodic boundary conditions on the estimated
        probability density. Default False, in which case no boundary
        conditions are imposed.

    num_posterior_samples: (int >= 0)
        Number of samples to draw from the Bayesian posterior. Restricted to
        0 <= num_posterior_samples <= MAX_NUM_POSTERIOR_SAMPLES.

    max_t_step: (float > 0)
        Upper bound on the amount by which the parameter ``t``
        in the DEFT algorithm is incremented when tracing the MAP curve.
        Default value is 1.0.

    tollerance: (float > 0)
        Sets the convergence criterion for the corrector algorithm used in
        tracing the MAP curve.

    resolution: (float > 0)
        The maximum geodesic distance allowed for neighboring points
        on the MAP curve.

    sample_only_at_l_star: (boolean)
        Specifies whether to let l vary when sampling from the Bayesian
        posterior.

    max_log_evidence_ratio_drop: (float > 0)
        If set, MAP curve tracing will terminate prematurely when
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop.

    evaluation_method_for_Z: (string)
        Method of evaluation of partition function Z. Possible values:
        'Lap'      : Laplace approximation (default).
        'Lap+Imp'  : Laplace approximation + importance sampling.
        'Lap+Fey'  : Laplace approximation + Feynman diagrams.

    num_samples_for_Z: (int >= 0)
        Number of posterior samples to use when evaluating the paritation
        function Z. Only has an affect when
        ``evaluation_method_for_Z = 'Lap+Imp'``.

    seed: (int)
        Seed provided to the random number generator before density estimation
        commences. For development purposes only.

    print_t: (bool)
        Whether to print the values of ``t`` while tracing the MAP curve.
        For development purposes only.

    attributes
    ----------
    grid:
        The grid points at which the probability density was be estimated.
        (1D np.array)

    grid_spacing:
        The distance between neighboring grid points.
        (float > 0)

    num_grid_points:
        The number of grid points used.
        (int)

    bounding_box:
        The boundaries of the data domain within which the probability density
        was be estimated. ([float, float])

    histogram:
        A histogram of the data using ``grid`` for the centers of each bin.
        (1D np.array)

    values:
        The values of the optimal (i.e., MAP) density at each grid point.
        (1D np.array)

    sample_values:
        The values of the posterior sampled densities at each grid point.
        The first index specifies grid points, the second posterior samples.
        (2D np.array)

    sample_weights:
        The importance weights corresponding to each posterior sample.
        (1D np.array)

    """

    @handle_errors
    def __init__(self,
                 data,
                 grid=None,
                 grid_spacing=None,
                 num_grid_points=None,
                 bounding_box=None,
                 alpha=3,
                 periodic=False,
                 num_posterior_samples=100,
                 max_t_step=1.0,
                 tolerance=1E-6,
                 resolution=0.1,
                 sample_only_at_l_star=False,
                 max_log_evidence_ratio_drop=20,
                 evaluation_method_for_Z='Lap',
                 num_samples_for_Z=1000,
                 seed=None,
                 print_t=False):

        # Record other inputs as class attributes
        self.alpha = alpha
        self.grid = grid
        self.grid_spacing = grid_spacing
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
        self.periodic = periodic
        self.Z_evaluation_method = evaluation_method_for_Z
        self.num_samples_for_Z = num_samples_for_Z
        self.max_t_step = max_t_step
        self.print_t = print_t
        self.tolerance = tolerance
        self.seed = seed
        self.resolution = resolution
        self.num_posterior_samples = num_posterior_samples
        self.sample_only_at_l_star = sample_only_at_l_star
        self.max_log_evidence_ratio_drop = max_log_evidence_ratio_drop
        self.data = data
        self.results = None

        # Validate inputs
        self._inputs_check()

        # clean input data
        self._clean_data()

        # Choose grid
        self._set_grid()

        # Fit to data
        self._run()

        # Save some results
        self.histogram = self.results.R
        self.maxent = self.results.M
        self.phi_star_values = self.results.phi_star

        # Compute evaluator for density
        self.density_func = DensityEvaluator(self.phi_star_values,
                                             self.grid,
                                             self.bounding_box)

        # Compute optimal density at grid points
        self.values = self.evaluate(self.grid)

        # If any posterior samples were taken
        if num_posterior_samples > 0:

            # Save sampled phi values and weights
            self.sample_field_values = self.results.phi_samples
            self.sample_weights = self.results.phi_weights

            # Compute evaluator for all posterior samples
            self.sample_density_funcs = [
                DensityEvaluator(field_values=self.sample_field_values[:, k],
                                 grid=self.grid,
                                 bounding_box=self.bounding_box)
                for k in range(self.num_posterior_samples)
            ]

            # Compute sampled values at grid points
            # These are NOT resampled
            self.sample_values = self.evaluate_samples(self.grid,
                                                       resample=False)

            # Compute effective sample size and efficiency
            self.effective_sample_size = np.sum(self.sample_weights)**2 \
                                        / np.sum(self.sample_weights**2)
            self.effective_sampling_efficiency = \
                self.effective_sample_size / self.num_posterior_samples

    @handle_errors
    def plot(self, ax=None,
             save_as=None,
             resample=True,
             figsize=(4, 4),
             fontsize=12,
             title='',
             xlabel='',
             tight_layout=False,
             show_now=True,
             show_map=True,
             map_color='blue',
             map_linewidth=2,
             map_alpha=1,
             num_posterior_samples=None,
             posterior_color='dodgerblue',
             posterior_linewidth=1,
             posterior_alpha=.2,
             show_histogram=True,
             histogram_color='orange',
             histogram_alpha=1,
             show_maxent=False,
             maxent_color='maroon',
             maxent_linewidth=1,
             maxent_alpha=1,
             backend='TkAgg'):
        """
        Plot the MAP density, the posterior sampled densities, and the
        data histogram.

        parameters
        ----------

        ax: (plt.Axes)
            A matplotlib axes object on which to draw. If None, one will be
            created

        save_as: (str)
            Name of file to save plot to. File type is determined by file
            extension.

        resample: (bool)
            If True, sampled densities will be ploted only after importance
            resampling.

        figsize: ([float, float])
            Figure size as (width, height) in inches.

        fontsize: (float)
            Size of font to use in plot annotation.

        title: (str)
            Plot title.

        xlabel: (str)
            Plot xlabel.

        tight_layout: (bool)
            Whether to call plt.tight_layout() after rendering graphics.

        show_now: (bool)
            Whether to show the plot immediately by calling plt.show().

        show_map: (bool)
            Whether to show the MAP density.

        map_color: (color spec)
            MAP density color.

        map_linewidth: (float)
            MAP density linewidth.

        map_alpha: (float)
            Map density opacity (between 0 and 1).

        num_posterior_samples: (int)
            Number of posterior samples to display. If this is greater than
            the number of posterior samples taken, all of the samples taken
            will be shown.

        posterior_color: (color spec)
            Sampled density color.

        posterior_linewidth: (float)
            Sampled density linewidth.

        posterior_alpha: (float)
            Sampled density opactity (between 0 and 1).

        show_histogram: (bool)
            Whether to show the (normalized) data histogram.

        histogram_color: (color spec)
            Face color of the data histogram.

        histogram_alpha: (float)
            Data histogram opacity (between 0 and 1).

        show_maxent: (bool)
            Whether to show the MaxEnt density estimate.

        maxent_color: (color spect)
            Line color of the MaxEnt density estimate.

        maxent_alpha: (float)
            MaxEnt opacity (between 0 and 1).

        backend: (str)
            Backend specification to send to sw.enable_graphics().

        returns
        -------

            None.

        """

        # check if matplotlib.pyplot is loaded. If not, load it carefully
        if 'matplotlib.pyplot' not in sys.modules:

            # First, enable graphics with the proper backend
            enable_graphics(backend=backend)

        # Make sure we have access to plt
        import matplotlib.pyplot as plt

        # If axes is not specified, create it and a corresponding figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            tight_layout = True

        # Plot histogram
        if show_histogram:
            ax.bar(self.grid,
                   self.histogram,
                   width=self.grid_spacing,
                   color=histogram_color,
                   alpha=histogram_alpha)

        # Plot maxent
        if show_maxent:
            ax.plot(self.grid,
                    self.maxent,
                    color=maxent_color,
                    linewidth=maxent_linewidth,
                    alpha=maxent_alpha)

        # Set number of posterior samples to plot
        if num_posterior_samples is None:
            num_posterior_samples = self.num_posterior_samples
        elif num_posterior_samples > self.num_posterior_samples:
            num_posterior_samples = self.num_posterior_samples

        # Plot posterior samples
        if num_posterior_samples > 0:
            sample_values = self.evaluate_samples(self.grid, resample=resample)
            ax.plot(self.grid,
                    sample_values[:, :num_posterior_samples],
                    color=posterior_color,
                    linewidth=posterior_linewidth,
                    alpha=posterior_alpha)

        # Plot best fit density
        if show_map:
            ax.plot(self.grid,
                    self.values,
                    color=map_color,
                    linewidth=map_linewidth,
                    alpha=map_alpha)

        # Style plot
        ax.set_xlim(self.bounding_box)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_yticks([])
        ax.tick_params('x', rotation=45, labelsize=fontsize)

        # Do not show interactive coordinates
        ax.format_coord = lambda x, y: ''

        # Do tight_layout if requested
        if tight_layout:
            plt.tight_layout()

        # Save figure if save_as is specified
        if save_as is not None:
            plt.draw()
            plt.savefig(save_as)

        # Show figure if show_now is True
        if show_now:
            plt.show()


    @handle_errors
    def evaluate(self, x):
        """
        Evaluate the optimal (i.e. MAP) density at the supplied value(s) of x.

        parameters
        ----------

        x: (number or list-like collection of numbers)
            The locations in the data domain at which to evaluate the MAP
            density.

        returns
        -------

        A float or 1D np.array representing the values of the MAP density at
        the specified locations.
        """

        # Clean input
        x_arr, is_number = clean_numerical_input(x)

        # Compute distribution values
        values = self.density_func.evaluate(x_arr)

        # If input is a single number, return a single number
        if is_number:
            values = values[0]

        # Return answer
        return values


    @handle_errors
    def evaluate_samples(self, x, resample=True):
        """
        Evaluate sampled densities at specified locations.

        parameters
        ----------

        x: (number or list-like collection of numbers)
            The locations in the data domain at which to evaluate sampled
            density.

        resample: (bool)
            Whether to use importance resampling, i.e., should the values
            returned be from the original samples (obtained using a Laplace
            approximated posterior) or should they be resampled to
            account for the deviation between the true Bayesian posterior
            and its Laplace approximation.

        returns
        -------

        A 1D np.array (if x is a number) or a 2D np.array (if x is list-like),
        representing the values of the posterior sampled densities at the
        specified locations. The first index corresponds to values in x, the
        second to sampled densities.
        """

        # Clean input
        x_arr, is_number = clean_numerical_input(x)

        # Check resample type
        check(isinstance(resample, bool),
              'type(resample) = %s. Must be bool.' % type(resample))

        # Make sure that posterior samples were taken
        check(self.num_posterior_samples > 0,
              'Cannot evaluate samples because no posterior samples'
              'have been computed.')

        assert(len(self.sample_density_funcs) == self.num_posterior_samples)

        # Evaluate all sampled densities at x
        values = np.array([d.evaluate(x_arr) for d
                           in self.sample_density_funcs]).T

        # If requested, resample columns of values array based on
        # sample weights
        if resample:
            probs = self.sample_weights / self.sample_weights.sum()
            old_cols = np.array(range(self.num_posterior_samples))
            new_cols = np.random.choice(old_cols,
                                        size=self.num_posterior_samples,
                                        replace=True,
                                        p=probs)
            values = values[:, new_cols]

        # If number was passed as input, return 1D np.array
        if is_number:
            values = values.ravel()

        return values

    @handle_errors
    def get_stats(self, use_weights=True,
                  show_samples=False):
        """
        Computes summary statistics for the estimated density

        parameters
        ----------

        show_samples: (bool)
            If True, summary stats are computed for each posterior sample.
            If False, summary stats are returned for the "star" estimate,
            the histogram, and the maxent estimate, along with the mean and
            RMSD values of these stats across posterior samples.

        use_weights: (bool)
            If True, mean and RMSD are computed using importance weights.

        returns
        -------

        df: (pd.DataFrame)
            A pandas data frame listing summary statistics for the estimated
            probability densities. These summary statistics include
            "entropy" (in bits), "mean", "variance", "skewness", and
            "kurtosis". If ``show_samples = False``, results will be shown for
            the best estimate, as well as mean and RMDS values across all
            samples. If ``show_samples = True``, results will be shown for
            each sample. A column showing column weights will also be included.
        """


        # Check inputs
        check(isinstance(use_weights, bool),
              'use_weights = %s; must be True or False.' % use_weights)
        check(isinstance(show_samples, bool),
              'show_samples = %s; must be True or False.' % show_samples)

        # Define a function for each summary statistic
        def entropy(Q):
            h = self.grid_spacing
            eps = 1E-10
            assert (all(Q >= 0))
            return np.sum(h * Q * np.log2(Q + eps))

        def mean(Q):
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * x)

        def variance(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 2)

        def skewness(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 3) / np.sum(
                h * Q * (x - mu) ** 2) ** (3 / 2)

        def kurtosis(Q):
            mu = mean(Q)
            x = self.grid
            h = self.grid_spacing
            return np.sum(h * Q * (x - mu) ** 4) / np.sum(
                h * Q * (x - mu) ** 2) ** 2

        # Index functions by their names and set these as columns
        col2func_dict = {'entropy': entropy,
                         'mean': mean,
                         'variance': variance,
                         'skewness': skewness,
                         'kurtosis': kurtosis}
        cols = list(col2func_dict.keys())
        if show_samples:
            cols += ['weight']

        # Create list of row names
        if show_samples:
            rows = ['sample %d' % n
                     for n in range(self.num_posterior_samples)]
        else:
            rows = ['star', 'histogram', 'maxent',
                    'posterior mean', 'posterior RMSD']

        # Initialize data frame
        df = pd.DataFrame(columns=cols, index=rows)

        # Set sample weights
        if use_weights:
            ws = self.sample_weights
        else:
            ws = np.ones(self.num_posterior_samples)

        # Fill in data frame column by column
        for col_num, col in enumerate(cols):

            # If listing weights, do so
            if col == 'weight':
                df.loc[:, col] = ws

            # If computing a summary statistic
            else:

                # Get summary statistic function
                func = col2func_dict[col]

                # Compute func value for each sample
                ys = np.zeros(self.num_posterior_samples)
                for n in range(self.num_posterior_samples):
                    ys[n] = func(self.sample_values[:, n])

                # If recording individual results for all samples, do so
                if show_samples:
                    df.loc[:, col] = ys

                # Otherwise, record individual entries
                else:
                    # Fill in func value for start density
                    df.loc['star', col] = func(self.values)

                    # Fill in func value for histogram
                    df.loc['histogram', col] = func(self.histogram)

                    # Fill in func value for maxent point
                    df.loc['maxent', col] = func(self.maxent)

                    # Record mean and rmsd values across samples
                    mu = np.sum(ys * ws) / np.sum(ws)
                    df.loc['posterior mean', col] = mu
                    df.loc['posterior RMSD', col] = np.sqrt(
                        np.sum(ws * (ys - mu) ** 2) / np.sum(ws))

        # Return data frame to user
        return df

    def _run(self):
        """
        Estimates the probability density from data using the DEFT algorithm.
        Also samples posterior densities
        """

        # Extract information from Deft1D object
        data = self.data
        G = self.num_grid_points
        h = self.grid_spacing
        alpha = self.alpha
        periodic = self.periodic
        Z_eval = self.Z_evaluation_method
        num_Z_samples = self.num_samples_for_Z
        DT_MAX = self.max_t_step
        print_t = self.print_t
        tollerance = self.tolerance
        resolution = self.resolution
        deft_seed = self.seed
        num_pt_samples = self.num_posterior_samples
        fix_t_at_t_star = self.sample_only_at_l_star
        max_log_evidence_ratio_drop = self.max_log_evidence_ratio_drop

        # Start clock
        clock = ConsumedTimeTimer()
        start_time = clock()

        # If deft_seed is specified, set it
        if not (deft_seed is None):
            np.random.seed(deft_seed)
        else:
            np.random.seed(None)

        # Create Laplacian
        laplacian_start_time = clock()
        if periodic:
            op_type = '1d_periodic'
        else:
            op_type = '1d_bilateral'
        Delta = laplacian.Laplacian(op_type, alpha, G)
        laplacian_compute_time = clock() - laplacian_start_time
        if print_t:
            print('Laplacian computed de novo in %f sec.'%laplacian_compute_time)

        # Get histogram counts and grid centers

        # Histogram based on bin centers
        counts, _ = np.histogram(data, self.bin_edges)
        N = sum(counts)

        # Make sure a sufficient number of bins are nonzero
        num_nonempty_bins = sum(counts > 0)
        check(num_nonempty_bins > self.alpha,
              'Histogram has %d nonempty bins; must be > %d.' %
              (num_nonempty_bins, self.alpha))

        # Compute initial t
        t_start = min(0.0, sp.log(N)-2.0*alpha*sp.log(alpha/h))
        if print_t:
            print('t_start = %0.2f' % t_start)

        # Do DEFT density estimation
        core_results = deft_core.run(counts, Delta, Z_eval, num_Z_samples,
                                     t_start, DT_MAX, print_t,
                                     tollerance, resolution, num_pt_samples,
                                     fix_t_at_t_star,
                                     max_log_evidence_ratio_drop)

        # Fill in results
        results = core_results # Get all results from deft_core

        # Normalize densities properly
        results.h = h
        results.L = G*h
        results.R /= h
        results.M /= h
        results.Q_star /= h
        results.l_star = h*(sp.exp(-results.t_star)*N)**(1/(2.*alpha))
        for p in results.map_curve.points:
            p.Q /= h
        if not (num_pt_samples == 0):
            results.Q_samples /= h
        results.Delta = Delta

        # Store results
        self.results = results


    def _inputs_check(self):
        """
        Check all inputs NOT having to do with the choice of grid
        :param self:
        :return: None
        """

        if self.grid_spacing is not None:

            # max_t_step is a number
            check(isinstance(self.grid_spacing, numbers.Real),
                  'type(grid_spacing) = %s; must be a number' %
                  type(self.grid_spacing))

            # grid_spacing is positive
            check(self.grid_spacing > 0,
                  'grid_spacing = %f; must be > 0.' % self.grid_spacing)

        if self.grid is not None:

            # grid is a list or np.array
            types = (list, np.ndarray, np.matrix)
            check(isinstance(self.grid, types),
                  'type(grid) = %s; must be a list or np.ndarray' %
                  type(self.grid))

            # cast grid as np.array as ints
            try:
                self.grid = np.array(self.grid).ravel().astype(float)
            except: # SHOULD BE MORE SPECIFIC
                raise ControlledError('Cannot cast grid as 1D np.array of floats.')

            # grid has appropriate number of points
            check(2*self.alpha <= len(self.grid) <= MAX_NUM_GRID_POINTS,
                  'len(grid) = %d; must have %d <= len(grid) <= %d.' %
                  (len(self.grid), 2*self.alpha, MAX_NUM_GRID_POINTS))

            # grid is ordered
            diffs = np.diff(self.grid)
            check(all(diffs > 0),
                  'grid is not monotonically increasing.')

            # grid is evenly spaced
            check(all(np.isclose(diffs, diffs.mean())),
                  'grid is not evenly spaced; grid spacing = %f +- %f' %
                  (diffs.mean(), diffs.std()))

        # alpha is int
        check(isinstance(self.alpha, int),
              'type(alpha) = %s; must be int.' % type(self.alpha))

        # alpha in range
        check(1 <= self.alpha <= 4,
              'alpha = %d; must have 1 <= alpha <= 4' % self.alpha)

        if self.num_grid_points is not None:

            # num_grid_points is an integer
            check(isinstance(self.num_grid_points, int),
                  'type(num_grid_points) = %s; must be int.' %
                  type(self.num_grid_points))

            # num_grid_points is in the right range
            check(2*self.alpha <= self.num_grid_points <= MAX_NUM_GRID_POINTS,
              'num_grid_points = %d; must have %d <= num_grid_poitns <= %d.' %
              (self.num_grid_points, 2*self.alpha, MAX_NUM_GRID_POINTS))

        # bounding_box
        if self.bounding_box is not None:

            # bounding_box is right type
            box_types = (list, tuple, np.ndarray)
            check(isinstance(self.bounding_box, box_types),
                  'type(bounding_box) = %s; must be one of %s' %
                  (type(self.bounding_box), box_types))

            # bounding_box has right length
            check(len(self.bounding_box)==2,
                  'len(bounding_box) = %d; must be %d' %
                  (len(self.bounding_box), 2))

            # bounding_box entries must be numbers
            check(isinstance(self.bounding_box[0], numbers.Real) and
                  isinstance(self.bounding_box[1], numbers.Real),
                  'bounding_box = %s; entries must be numbers' %
                  repr(self.bounding_box))

            # bounding_box entries must be sorted
            check(self.bounding_box[0] < self.bounding_box[1],
                  'bounding_box = %s; entries must be sorted' %
                  repr(self.bounding_box))

            # reset bounding_box as tuple
            self.bounding_box = (float(self.bounding_box[0]),
                                 float(self.bounding_box[1]))

        # periodic is bool
        check(isinstance(self.periodic, bool),
              'type(periodic) = %s; must be bool' % type(self.periodic))

        # evaluation_method_for_Z is valid
        Z_evals = ['Lap', 'Lap+Imp', 'Lap+Fey']
        check(self.Z_evaluation_method in Z_evals,
              'Z_eval = %s; must be in %s' %
              (self.Z_evaluation_method, Z_evals))

        # num_samples_for_Z is an integer
        check(isinstance(self.num_samples_for_Z, numbers.Integral),
              'type(self.num_samples_for_Z) = %s; ' %
              type(self.num_samples_for_Z) +
              'must be integer.')
        self.num_samples_for_Z = int(self.num_samples_for_Z)

        # num_samples_for_Z is in range
        check(0 <= self.num_samples_for_Z <= MAX_NUM_SAMPLES_FOR_Z,
              'self.num_samples_for_Z = %d; ' % self.num_samples_for_Z +
              ' must satisfy 0 <= num_samples_for_Z <= %d.' %
               MAX_NUM_SAMPLES_FOR_Z)

        # max_t_step is a number
        check(isinstance(self.max_t_step, numbers.Real),
              'type(max_t_step) = %s; must be a number' %
              type(self.max_t_step))

        # max_t_step is positive
        check(self.max_t_step > 0,
              'maxt_t_step = %f; must be > 0.' % self.max_t_step)

        # print_t is bool
        check(isinstance(self.print_t,bool),
              'type(print_t) = %s; must be bool.' % type(self.print_t))

        # tolerance is float
        check(isinstance(self.tolerance, numbers.Real),
              'type(tolerance) = %s; must be number' % type(self.tolerance))

        # tolerance is positive
        check(self.tolerance > 0,
              'tolerance = %f; must be > 0' % self.tolerance)

        # resolution is number
        check(isinstance(self.resolution, numbers.Real),
              'type(resolution) = %s; must be number' % type(self.resolution))

        # resolution is positive
        check(self.resolution > 0,
              'resolution = %f; must be > 0' % self.resolution)

        if self.seed is not None:

            # seed is int
            check(isinstance(self.seed, int),
                  'type(seed) = %s; must be int' % type(self.seed))

            # seed is in range
            check(0 <= self.seed <= 2**32 - 1,
                  'seed = %d; must have 0 <= seed <= 2**32 - 1' % self.seed)

        # sample_only_at_l_star is bool
        check(isinstance(self.sample_only_at_l_star, bool),
              'type(sample_only_at_l_star) = %s; must be bool.' %
              type(self.sample_only_at_l_star))

        # num_posterior_samples is int
        check(isinstance(self.num_posterior_samples, numbers.Integral),
              'type(num_posterior_samples) = %s; must be integer' %
              type(self.num_posterior_samples))
        self.num_posterior_samples = int(self.num_posterior_samples)


        # num_posterior_samples is nonnegative
        check(0 <= self.num_posterior_samples <= MAX_NUM_POSTERIOR_SAMPLES,
              'num_posterior_samples = %f; need '%self.num_posterior_samples +
              '0 <= num_posterior_samples <= %d.' %MAX_NUM_POSTERIOR_SAMPLES)

        # max_log_evidence_ratio_drop is number
        check(isinstance(self.max_log_evidence_ratio_drop, numbers.Real),
              'type(max_log_evidence_ratio_drop) = %s; must be number' %
              type(self.max_log_evidence_ratio_drop))

        # max_log_evidence_ratio_drop is positive
        check(self.max_log_evidence_ratio_drop > 0,
              'max_log_evidence_ratio_drop = %f; must be > 0' %
              self.max_log_evidence_ratio_drop)


    def _clean_data(self):
        """
        Sanitize the assigned data
        :param: self
        :return: None
        """
        data = self.data

        # if data is a list-like, convert to 1D np.array
        if isinstance(data, LISTLIKE):
            data = np.array(data).ravel()
        elif isinstance(data, set):
            data = np.array(list(data)).ravel()
        else:
            raise ControlledError(
                "Error: could not cast data into an np.array")

        # Check that entries are numbers
        check(all([isinstance(n, numbers.Real) for n in data]),
              'not all entries in data are real numbers')

        # Cast as 1D np.array of floats
        data = data.astype(float)

        # Keep only finite numbers
        data = data[np.isfinite(data)]


        try:
            if not (len(data) > 0):
                raise ControlledError(
                    'Input check failed, data must have length > 0: data = %s' % data)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        try:
            data_spread = max(data) - min(data)
            if not np.isfinite(data_spread):
                raise ControlledError(
                    'Input check failed. Data[max]-Data[min] is not finite: Data spread = %s' % data_spread)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        try:
            if not (data_spread > 0):
                raise ControlledError(
                    'Input check failed. Data[max]-Data[min] must be > 0: data_spread = %s' % data_spread)
        except ControlledError as e:
            print(e)
            sys.exit(1)

        # Set cleaned data
        self.data = data


    def _set_grid(self):
        """
        Sets the grid based on user input
        """

        data = self.data
        grid = self.grid
        grid_spacing = self.grid_spacing
        num_grid_points = self.num_grid_points
        bounding_box = self.bounding_box
        alpha = self.alpha

        # If grid is specified
        if grid is not None:

            # Check and set number of grid points
            num_grid_points = len(grid)
            assert(num_grid_points >= 2*alpha)

            # Check and set grid spacing
            diffs = np.diff(grid)
            grid_spacing = diffs.mean()
            assert (grid_spacing > 0)
            assert (all(np.isclose(diffs, grid_spacing)))

            # Check and set grid bounds
            grid_padding = grid_spacing / 2
            lower_bound = grid[0] - grid_padding
            upper_bound = grid[-1] + grid_padding
            bounding_box = np.array([lower_bound, upper_bound])
            box_size = upper_bound - lower_bound

        # If grid is not specified
        if grid is None:

            ### First, set bounding box ###

            # If bounding box is specified, use that.
            if bounding_box is not None:
                assert bounding_box[0] < bounding_box[1]
                lower_bound = bounding_box[0]
                upper_bound = bounding_box[1]
                box_size = upper_bound - lower_bound


            # Otherwise set bounding box based on data
            else:
                assert isinstance(data, np.ndarray)
                assert all(np.isfinite(data))
                assert min(data) < max(data)

                # Choose bounding box to encapsulate all data, with extra room
                data_max = max(data)
                data_min = min(data)
                data_span = data_max - data_min
                lower_bound = data_min - .2 * data_span
                upper_bound = data_max + .2 * data_span

                # Autoadjust lower bound
                if data_min >= 0 and lower_bound < 0:
                    lower_bound = 0

                # Autoadjust upper bound
                if data_max <= 0 and upper_bound > 0:
                    upper_bound = 0
                if data_max <= 1 and upper_bound > 1:
                    upper_bound = 1
                if data_max <= 100 and upper_bound > 100:
                    upper_bound = 100

                # Extend bounding box outward a little for numerical safety
                lower_bound -= SMALL_NUM*data_span
                upper_bound += SMALL_NUM*data_span
                box_size = upper_bound - lower_bound

                # Set bounding box
                bounding_box = np.array([lower_bound, upper_bound])

            ### Next, define grid based on bounding box ###

            # If grid_spacing is specified
            if (grid_spacing is not None):
                assert isinstance(grid_spacing, float)
                assert np.isfinite(grid_spacing)
                assert grid_spacing > 0

                # Set number of grid points
                num_grid_points = np.floor(box_size/grid_spacing).astype(int)

                # Check num_grid_points isn't too small
                check(2*self.alpha <= num_grid_points,
                      'Using grid_spacing = %f ' % grid_spacing +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too small. Reduce grid_spacing or do not set.')

                # Check num_grid_points isn't too large
                check(num_grid_points <= MAX_NUM_GRID_POINTS,
                      'Using grid_spacing = %f ' % grid_spacing +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too big. Increase grid_spacing or do not set.')

                # Define grid padding
                # Note: grid_spacing/2 <= grid_padding < grid_spacing
                grid_padding = (box_size - (num_grid_points-1)*grid_spacing)/2
                assert (grid_spacing/2 <= grid_padding < grid_spacing)

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM), # For safety
                                   num_grid_points)

            # Otherwise, if num_grid_points is specified
            elif (num_grid_points is not None):
                assert isinstance(num_grid_points, int)
                assert 2*alpha <= num_grid_points <= MAX_NUM_GRID_POINTS

                # Set grid spacing
                grid_spacing = box_size / num_grid_points

                # Define grid padding
                grid_padding = grid_spacing/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM), # For safety
                                   num_grid_points)

            # Otherwise, set grid_spacing and num_grid_points based on data
            else:
                assert isinstance(data, np.ndarray)
                assert all(np.isfinite(data))
                assert min(data) < max(data)

                # Compute default grid spacing
                default_grid_spacing = box_size/DEFAULT_NUM_GRID_POINTS

                # Set minimum number of grid points
                min_num_grid_points = 2 * alpha

                # Set minimum grid spacing
                data.sort()
                diffs = np.diff(data)
                min_grid_spacing = min(diffs[diffs > 0])
                min_grid_spacing = min(min_grid_spacing,
                                       box_size/min_num_grid_points)

                # Set grid_spacing
                grid_spacing = max(min_grid_spacing, default_grid_spacing)

                # Set number of grid points
                num_grid_points = np.floor(box_size/grid_spacing).astype(int)

                # Set grid padding
                grid_padding = grid_spacing/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM),  # For safety
                                   num_grid_points)

        # Set final grid
        self.grid = grid
        self.grid_spacing = grid_spacing
        self.grid_padding = grid_padding
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.box_size = box_size

        # Make sure that the final number of gridpoints is ok.
        check(2 * self.alpha <= self.num_grid_points <= MAX_NUM_GRID_POINTS,
              'After setting grid, we find that num_grid_points = %d; must have %d <= len(grid) <= %d. ' %
              (self.num_grid_points, 2*self.alpha, MAX_NUM_GRID_POINTS) +
              'Something is wrong with input values of grid, grid_spacing, num_grid_points, or bounding_box.')

        # Set bin edges
        self.bin_edges = np.concatenate(([lower_bound],
                                         grid[:-1]+grid_spacing/2,
                                         [upper_bound]))

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class DensityEvaluator:
    """
    A probability density that can be evaluated at anywhere

    Parameters
    ----------

    field_values: (1D np.array)

        The values of the field used to computed this density.

    grid: (1D np.array)

        The grid points at which the field values are defined. Must be the same
        the same shape as field.

    Attributes
    ----------

    field_values:
        See above.

    grid:
        See above.

    grid_spacing: (float)
        The spacing between neighboring gridpoints.

    values: (1D np.array)
        The values of the probability density at all grid points.

    bounding_box:
        The domain in which the density is nonzero.

    Z:
        The normalization constant used to convert the field to a density.

    """

    def __init__(self, field_values, grid, bounding_box,
                 interpolation_method='cubic'):

        # Make sure grid and field are the same size
        self.field_values = field_values
        self.grid = grid
        self.bounding_box = bounding_box

        # Compute grid spacing
        self.grid_spacing = grid[1]-grid[0]

        # Compute normalization constant
        self.Z = np.sum(self.grid_spacing * np.exp(-self.field_values))

        # Interpolate using extended grid and extended phi
        self.field_func = interpolate.interp1d(self.grid,
                                               self.field_values,
                                               kind=interpolation_method,
                                               bounds_error=False,
                                               fill_value='extrapolate',
                                               assume_sorted=True)

        # Compute density values at supplied grid points
        self.values = self.evaluate(xs=self.grid)


    def evaluate(self, xs):
        """
        Evaluates the probability density at specified positions.

        Note: self(xs) can be used instead of self.evaluate(xs).

        Parameters
        ----------

        xs: (np.array)
            Locations at which to evaluate the density.

        Returns
        -------

        (np.array)
            Values of the density at the specified positions. Values at
            positions outside the bounding box are evaluated to zero.

        """

        values = np.exp(-self.field_func(xs)) / self.Z
        zero_indices = (xs < self.bounding_box[0]) | \
                       (xs > self.bounding_box[1])
        values[zero_indices] = 0.0
        return values

    def __call__(self, *args, **kwargs):
        """
        Same as evaluate()
        """

        return self.evaluate(*args, **kwargs)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



# Enable plotting
from src.utils import enable_graphics, check, ControlledError

# # Classes that have yet to be written
# class Density2DEstimator:
#     """
#     Future class for density estimation in a two dimensional area.
#     """
#     pass
#
# class JointDensityEstimator:
#     """
#     Future class for estimating the joint distribution between two
#     univariate quantities.
#     """
#     pass
#
# class SurvivalCurveEstimator:
#     """
#     Future class for computing simple survival curves
#     """
#     pass
#
# class ProportionalHazardsEstimator:
#     """
#     Future class for computing proportional hazards models
#     """
#     pass
#
# class GeneralizedHazardsEstimator:
#     """
#     Future class for computing generalized hazards models
#     """
#     pass
#
# class IntervalCensoredSurvivalEstimator:
#     """
#     Future class for computing interval-censored survival curves
#     """
#

# try_me functions
def demo(example='real_data'):
    """
    Performs a demonstration of suftware.

    Parameters
    ----------

    example: (str)
        A string specifying which demo to run. Must be 'real_data',
        'simulated_data', or 'custom_data'.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)

    example_dict = {
        'custom_data': 'docs/example_custom.py',
        'simulated_data': 'docs/example_wide.py',
        'real_data': 'docs/example_alcohol.py'
    }

    check(example in example_dict,
          'example = %s is not valid. Must be one of %s'%\
          (example, example_dict.keys()))

    file_name = '%s/%s'%(example_dir, example_dict[example])
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s'%\
              (file_name, line, content, line))
    exec(open(file_name).read())
