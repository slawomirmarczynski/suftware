#!/usr/local/bin/python -W ignore
import os.path
import re
import scipy as sp
import numpy as np
import sys
import time
from scipy.interpolate import interp1d

# Import deft-related code
from deft_code import deft_core
from deft_code import utils
from deft_code import laplacian
from deft_code import maxent

# Import error handling
from deft_code.utils import DeftError

class Results(): pass;


#
# The main DEFT algorithm in 1D.
#

def run(obj):

    # Extract information from Deft1D object
    data = obj.data
    G = obj.num_grid_points
    alpha = obj.alpha
    bbox = obj.bounding_box
    periodic = obj.periodic
    Z_eval = obj.Z_evaluation_method
    num_Z_samples = obj.num_samples_for_Z
    DT_MAX = obj.max_t_step
    print_t = obj.print_t
    tollerance = obj.tolerance
    resolution = obj.resolution
    deft_seed = obj.seed
    num_pt_samples = obj.num_posterior_samples
    fix_t_at_t_star = obj.sample_only_at_l_star
    max_log_evidence_ratio_drop = obj.max_log_evidence_ratio_drop

    # Start clock
    start_time = time.clock()

    # If deft_seed is specified, set it
    if not (deft_seed is None):
        np.random.seed(deft_seed)
    else:
        np.random.seed(None)
        
    # Create Laplacian
    laplacian_start_time = time.clock()
    if periodic:
        op_type = '1d_periodic'
    else:
        op_type = '1d_bilateral'
    Delta = laplacian.Laplacian(op_type, alpha, G)
    laplacian_compute_time = time.clock() - laplacian_start_time
    if print_t:
        print('Laplacian computed de novo in %f sec.'%laplacian_compute_time)

    # Get histogram counts and grid centers
    counts, bin_centers = utils.histogram_counts_1d(data, G, bbox)
    N = sum(counts)

    # Get other information about grid
    bbox, h, bin_edges = utils.grid_info_from_bin_centers_1d(bin_centers)

    # Compute initial t
    t_start = min(0.0, sp.log(N)-2.0*alpha*sp.log(alpha/h))
    if t_start < -10.0:
        t_start /= 2
    #print('t_start = %.2f' % t_start)
    if print_t:
        print('t_start = %0.2f' % t_start)
    
    # Do DEFT density estimation
    core_results = deft_core.run(counts, Delta, Z_eval, num_Z_samples, t_start, DT_MAX, print_t,
                                 tollerance, resolution, num_pt_samples, fix_t_at_t_star,max_log_evidence_ratio_drop)

    # Fill in results
    copy_start_time = time.clock()
    results = core_results # Get all results from deft_core

    # Normalize densities properly
    results.h = h 
    results.L = G*h
    results.R /= h
    results.Q_star /= h
    results.l_star = h*(sp.exp(-results.t_star)*N)**(1/(2.*alpha))
    for p in results.map_curve.points:
        p.Q /= h
    if not (num_pt_samples == 0):
        results.Q_samples /= h

    # Get 1D-specific information
    results.bin_centers = bin_centers
    results.bin_edges = bin_edges
    results.periodic = periodic
    results.alpha = alpha
    results.bbox = bbox
    results.Delta = Delta
    copy_compute_time = time.clock() - copy_start_time

    # Create interpolated phi_star and Q_star. Need to extend grid to boundaries first
    extended_xgrid = sp.zeros(G+2)
    extended_xgrid[1:-1] = bin_centers
    extended_xgrid[0] = bbox[0] - h/2
    extended_xgrid[-1] = bbox[1] + h/2
    
    extended_phi_star = sp.zeros(G+2)
    extended_phi_star[1:-1] = results.phi_star
    extended_phi_star[0] = results.phi_star[0]
    extended_phi_star[-1] = results.phi_star[-1]

    phi_star_func = interp1d(extended_xgrid, extended_phi_star, kind='cubic')
    Z = sp.sum(h*sp.exp(-results.phi_star))
    #Q_star_func = lambda(x): sp.exp(-phi_star_func(x))/Z
    Q_star_func = lambda x: sp.exp(-phi_star_func(x)) / Z
    results.Q_star_func = Q_star_func

    # XXX REMOVE
    # Compute differential entropy in bits
    entropy_start_time = time.clock()
    if not (num_pt_samples == 0):
        entropies = np.zeros(num_pt_samples)
        for i in range(results.Q_samples.shape[1]):
            Q = results.Q_samples[:,i].ravel()
            entropy = -sp.sum(h*Q*sp.log2(Q + utils.TINY_FLOAT64))
            #for j in range(G):
            #    entropy += -results.h*Q[j]*sp.log2(Q[j] + utils.TINY_FLOAT64)
            entropies[i] = entropy

        # Compute mean and variance of the differential entropy
        results.entropies = entropies
        results.e_mean = np.mean(entropies)
        results.e_std = np.std(entropies)
        results.entropy_compute_time = time.clock() - entropy_start_time
    # XXX

    # Record execution time
    results.copy_compute_time = copy_compute_time
    results.laplacian_compute_time = laplacian_compute_time
    results.deft_1d_compute_time = time.clock()-start_time

    return results
