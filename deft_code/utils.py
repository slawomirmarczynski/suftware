from __future__ import division
import scipy as sp
import numpy as np
import pdb
import sys

from numpy.polynomial.legendre import legval, legval2d

# A very small floating point number, used to prevent taking logs of 0
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
TINY_FLOAT32 = sp.finfo(sp.float32).tiny
PHI_MIN = -100
PHI_MAX = 100
PHI_STD_REG = 100.0
SMALL_POS_NUM = sys.float_info.epsilon

# This is useful for testing whether something is a number
NUMBER = (int, float, int)
ARRAY = (np.ndarray, list)

def safe_exp(v):
    ''' Returns a value for np.exp(v) without underflow and overflow errors '''
    assert isinstance(v,ARRAY+NUMBER),\
        'Error: trying to exponentiate something of type %s.'%str(type(v))
    if isinstance(v,ARRAY):
        v = np.array(v).copy()
        assert not any(v > PHI_MAX),\
            'Error: trying to exponentiate numbers that are too large'
        v[v<PHI_MIN] = PHI_MIN
    elif isinstance(v,NUMBER):
        assert not v > PHI_MAX,\
            'Error: trying to exponentiate numbers that are too large'
        if v < PHI_MIN:
            v = PHI_MIN
    return np.exp(v)

# Evaluate geodesic distance
def geo_dist(P,Q):

    # Make sure these are valid distributions
    assert all(np.isreal(P))
    assert all(P >= 0)
    assert any(P > 0)
    assert all(np.isreal(Q))
    assert all(Q >= 0)
    assert any(Q > 0)

    # Enforce proper normalization
    P_prob = P/sp.sum(P)
    Q_prob = Q/sp.sum(Q)

    # Return geodistance. arccos can behave badly if argument
    # is too close to one, so prepare for this
    try:
        dist = 2*sp.arccos(sp.sum(sp.sqrt(P_prob*Q_prob)))
        assert np.isreal(dist)
    except:
        if sp.sum(sp.sqrt(P_prob*Q_prob)) > 1 - TINY_FLOAT32:
            dist = 0
        else:
            raise
    return dist

# Convert field to non-normalized probabiltiy distribution
def field_to_quasiprob(raw_phi):
    phi = np.copy(raw_phi)
    G = len(phi)

    # Make sure phi is real
    # If not, there is nothing we can do. Abort
    assert all(np.isreal(phi))

    # If any values of phi are too large
    # set equal to upper bound
    #if any(phi > PHI_MAX):
    #    phi[phi > PHI_MAX] = PHI_MAX

    # If any values of phi are too samll, abort
    assert all(phi > PHI_MIN)

    # Compute quasiQ
    quasiQ = sp.exp(-phi)/(1.*G)

    # Make sure Q is finit
    assert all(np.isfinite(quasiQ))

    # Return quasiprob
    return quasiQ

# Convert field to normalized probability distribution
def field_to_prob(raw_phi):
    phi = np.copy(raw_phi)
    G = len(phi)

    # Make sure phi is real
    # If not, there is nothing we can do. Abort
    assert all(np.isreal(phi))

    # Relevel phi
    # NOTE: CHANGES PHI!
    phi -= min(phi)

    # If any values of phi are too large
    # set equal to upper bound
    #if any(phi > PHI_MAX):
    #    phi[phi > PHI_MAX] = PHI_MAX
    # Compute quasiQ
    denom = sp.sum(sp.exp(-phi))
    Q = sp.exp(-phi)/denom

    # Make sure Q is finit
    assert all(np.isfinite(Q))

    # Return probability
    return Q

# Convert probability distribution to field
def prob_to_field(Q):
    G = len(Q)
    assert(all(np.isreal(Q)))
    assert(all(Q >= 0))
    phi = -sp.log(G*Q + TINY_FLOAT64)
    assert(all(np.isreal(phi)))
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
    G = len(bin_centers)
    bin_centers = bin_edges[:-1]+h/2.
    return bbox, h, bin_centers

def grid_info_from_bbox_and_G(bbox, G):
    bin_edges = np.linspace(bbox[0], bbox[1], num=G+1, endpoint=True)
    h = bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges[:-1]+h/2.
    return h, bin_centers, bin_edges


# Make a 1d histogram. Bounding box is optional
def histogram_counts_1d(data, G, bbox='auto', normalized=False):

    # initializing empty cropped data to avoid referencing before initialization errors
    cropped_data = np.array([])

    # If setting interval automatically
    if bbox=='auto':
        # Make sure there is actually data present
        assert len(data) > 0,\
            ''' Error: cannot set bbox automatically when fewer than 2 
            distinct data points are proivded. Here, only %d distinct 
            data points were provided'''%len(set(data))

        # if data contains metadata
        if type(data) is tuple:
            # Get interval spanned by data
            data_interval = max(data[0])-min(data[0])

            # Make sure that this interval is finite
            assert np.isfinite(data_interval)

            # Make sure the data actually have some spread
            assert data_interval > 0

            # Set bbox
            bbox = []
            bbox.append(min(data[0]) - data_interval*0.2)
            bbox.append(max(data[0]) + data_interval*0.2)

            # Make sure bbox is valid
            assert isinstance(bbox[0], NUMBER)
            assert isinstance(bbox[1], NUMBER)
            assert bbox[0] < bbox[1]

            # Crop data to bounding box: if any data are lower or greater than bounding box, ignore them
            # indices = (data > bbox[0]) & (data < bbox[1])
            indices = np.where((data[0] > bbox[0]) & (data[0] < bbox[1]))
            cropped_data = data[0][indices]

        # if data contains just raw data without metadata
        elif type(data) is np.ndarray:
            # Get interval spanned by data
            data_interval = max(data) - min(data)

            # Set bbox
            bbox = []

            bbox.append(min(data) - data_interval * 0.2)
            bbox.append(max(data) + data_interval * 0.2)

            indices = (data > bbox[0]) & (data < bbox[1])
            cropped_data = data[indices]

            # Make sure bbox is valid
            assert isinstance(bbox[0], NUMBER)
            assert isinstance(bbox[1], NUMBER)
            assert bbox[0] < bbox[1]

            # Crop data to bounding box: if any data are lower or greater than bounding box, ignore them
            indices = (data > bbox[0]) & (data < bbox[1])
            cropped_data = data[indices]

    # Get grid info from bbox and G
    h, bin_centers, bin_edges = grid_info_from_bbox_and_G(bbox, G)

    # Get counts in each bin
    #counts, _ = np.histogram(data[0], bins=bin_edges, density=False)
    if len(cropped_data) is not 0:
        counts, _ = np.histogram(cropped_data, bins=bin_edges, density=False)
    elif type(data) is tuple:
        counts, _ = np.histogram(data[0], bins=bin_edges, density=False)
    else:
        counts, _ = np.histogram(data, bins=bin_edges, density=False)

    if normalized:
        hist = 1.*counts/np.sum(h*counts)
    else:
        hist = counts

    # Return the number of counts, the bin centers, and the bbox
    return hist, bin_centers

# Make a histogram
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
    assert( len(v2)==G )
    return sp.sum(v1r*v2r*h/(1.*G))

# Comptues a norm with my normalization
def norm(v,h=1.0):
    v_cc = np.conj(v)
    return sp.sqrt(dot(v,v_cc,h))

# Normalizes vectors (stored as columns of a 2D numpy array)
def normalize(vectors, grid_spacing=1.0):
    """ Normalizes vectors stored as columns of a 2D numpy array """
    K = vectors.shape[1] # number of vectors
    G = vectors.shape[0] # length of each vector

    # Set volume element h. This takes a little consideration
    if (isinstance(grid_spacing,NUMBER)):
        h = grid_spacing
    elif (isinstance(grid_spacing,ARRAY)):
        grid_spacing = sp.array(grid_spacing)
        h = sp.prod(grid_spacing)
    else:
        print ('ERROR: what kind of thing is grid_spacing?')
        print (type(grid_spacing))
        raise

    assert (h > 0)

    norm_vectors = sp.zeros([G,K])
    for i in range(K):

        # Extract v from list of vectors
        v = vectors[:,i]

        # Flip sign of v so that last element is nonnegative
        if (v[-1] < 0):
            v = -v

        # Normalize v and save in norm_vectors
        norm_vectors[:,i] = v/norm(v)

    # Return array with normalized vectors along the columns
    return norm_vectors

# Construct an orthonormal basis of order alpha from 1d legendre polynomials
def legendre_basis_1d(G, alpha, grid_spacing=1.0):

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
    xs,ys = np.meshgrid(x_grid,y_grid)

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