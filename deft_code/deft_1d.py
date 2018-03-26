#!/usr/local/bin/python -W ignore
import scipy as sp
import numpy as np
import sys
import time
from scipy.interpolate import interp1d
import pdb

# Import deft-related code
from deft_code import deft_core
from deft_code import utils
from deft_code import laplacian

from deft_code.supplements import inputs_check
from deft_code.supplements import clean_data
from deft_code.utils import DeftError

from deft_code.density_1d import Density1D
from deft_code.field_1d import Field1D

class Deft1D:
    """This class will serve as the interface for running
    deft1d

    parameters
    ----------
    data: (np.array)
        User input data for which Deft1D will estimate the density.
    num_grid_points: (int)
        Number of grid points.
    grid: (np.array)
        Locations of grid points. Grid points must be evenly spaced in
        ascending order.
    whisker_length: (float)
        The number of interquartile ranges above the 3rd quartile and below the
        1st quartile at which to position the upper and lower bounds of the
        bounding box.
    alpha: (int)
        Smoothness parameter. Represents the order of the
        derivative in the action.
    bounding_box: ([float,float])
        Bounding box for density estimation.
    periodic: (boolean)
        Enforce periodic boundary conditions via True or False.
    Z_evaluation_method: (string)
        Method of evaluation of partition function. Possible values:
        'Lap'      : Laplace approximation (default).
        'Lap+Imp'  : Laplace approximation + importance sampling.
        'Lap+Fey'  : Laplace approximation + Feynman diagrams.
    num_samples_for_Z: (int)
        *** Note *** This parameter works only when Z_evaluation_method is 'Lap+Imp'.
        Number of samples for the evaluation of the partition function.
        More samples will help to evaluate a better density. More samples
        will also make calculation slower. num_samples_for_Z = 0 means the Laplace
        approximation for the evaluation of the partition function is used.
    resolution: (float > 0)
        Specifies max distance between neighboring points on the MAP curve.
    seed: (int)
        Specify random seed for evaluation of the partition function
        and for the posterior sampling.
    max_t_step: (float > 0)
        Maximum t step size on the MAP curve.
    max_log_evidence_ratio_drop: (float > 0)
        Stop criterion for traversing the MAP curve; deft stops when:
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop
    tolerance: (float > 0)
        Value which species convergence of phi.
    num_posterior_samples: (int >= 0)
        Number of posterior samples.
    sample_only_at_l_star: (boolean)
        If True : posterior samples drawn at l_star.
        If False: posterior sampling done among l near l_star.

    attributes
    ----------
    *** Note ***: all parameters are attributes except for results.
    results: (dict)
        Contains the results of deft.

    methods
    -------
    run():
        Runs the DEFT 1D algorithm.
    get_results():
        Transforms the results object passed by run() into a dictionary, or
        returns a single specified attribute of the results object.
    get_Q_samples():
        Returns a list of sampled Qs in functionalized form.
    eval_samples():
        Evaluates sampled Qs on specified x values.
    """

    def __init__(self,
                 data,
                 num_grid_points=100,
                 bounding_box=None,
                 grid=None,
                 whisker_length=3,
                 alpha=3,
                 periodic=False,
                 Z_evaluation_method='Lap',
                 num_samples_for_Z=1e5,
                 max_t_step=1.0,
                 print_t=False,
                 tolerance=1E-6,
                 resolution=0.1,
                 seed=None,
                 num_posterior_samples=100,
                 sample_only_at_l_star=False,
                 max_log_evidence_ratio_drop=20):

        # Record inputs in class attributes
        self.num_grid_points = num_grid_points
        self.alpha = alpha
        self.bounding_box = bounding_box
        self.grid = grid
        self.periodic = periodic
        self.Z_evaluation_method = Z_evaluation_method
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

        # clean input data
        self.data, self.min_h = clean_data(data)

        # Validate inputs
        inputs_check(self)

        # If grid is specified, use that
        if grid is not None:
            assert (len(grid) >= 2)
            grid = np.array(grid).copy()
            grid = np.linspace(grid.min(), grid.max(), len(grid))
            grid_spacing = grid[1]-grid[0]
            bounding_box = [grid[0] - grid_spacing/2,
                            grid[-1] + grid_spacing/2]
            num_grid_points = len(grid)

        # Otherwise, if bounding box is specified, use that
        elif bounding_box is not None:
            assert(len(bounding_box) == 2)
            assert(bounding_box[0] < bounding_box[1])
            assert(num_grid_points > 1)
            grid_spacing = (bounding_box[1] - bounding_box[0])/num_grid_points
            grid = np.linspace(bounding_box[0] + grid_spacing/2,
                               bounding_box[1] - grid_spacing/2,
                               num_grid_points)



        # Otherwise, choose bbox to follow wiskers in a box plot, i.e.
        # lower quartile - whisker_length*IQR to
        # upper quartile + whisker_length*IQR
        # then adjust range as appropriate
        else:
            lower_quartile = np.percentile(data, 25)
            upper_quartile = np.percentile(data, 75)
            iqr = upper_quartile - lower_quartile
            lower_bound = lower_quartile - whisker_length * iqr
            upper_bound = upper_quartile + whisker_length * iqr

            # Autoadjust lower bound
            if all(data >= 0) and lower_bound < 0:
                lower_bound = 0

            # Autoadjust upper bound
            if all(data <= 0) and upper_bound > 0:
                upper_bound = 0
            if all(data <= 1) and upper_bound > 1:
                upper_bound = 1
            if all(data <= 100) and upper_bound > 100:
                upper_bound = 100

            bounding_box = [lower_bound-(1E-6)*iqr, upper_bound+(1E-6)*iqr]
            assert(num_grid_points > 1)
            grid_spacing = (bounding_box[1] - bounding_box[0])/num_grid_points
            grid = np.linspace(bounding_box[0] + grid_spacing / 2,
                               bounding_box[1] - grid_spacing / 2,
                               num_grid_points)

        # Set final grid
        self.grid = grid
        self.bounding_box = bounding_box
        self.grid_spacing = grid_spacing
        self.num_grid_points = num_grid_points

        # Fit to data
        self.results = self.run()

        # Save some results
        self.results_dict = self.get_results()
        self.histogram = self.results_dict['R']

        self.phi_star = Field1D(self.results_dict['phi_star'],
                                self.grid,
                                self.bounding_box)
        self.Q_star = Density1D(self.phi_star)
        self.evaluate = self.Q_star.evaluate

        self.values = self.evaluate(self.grid)
        self.sample_values = self.eval_samples(self.grid)


    def get_results(self, key=None):
        """
        Returns a dictionary whose keys access the attributes of the
        results object returned by deft_1d.run()

        [DOCUMENT]
        """
        if self.results is not None and key is None:
            # return the dictionary containing results if no key provided
            return self.results.__dict__
        elif self.results is not None and key is not None:
            try:
                return self.results.__dict__.get(key)
                #return self.results.__getattribute__(key)
            except AttributeError as e:
                print("Get results:",e)
        else:
            print("Get Results: Deft results are none. Please run fit first.")


    def get_Q_samples(self, importance_resampling=True):
        """
        Produces a set of sampled Q distributions. By default these are chosen
        using importance resampling.

        [DOCUMENT]
        """

        # ensure parameters are legal
        if self.results is not None and self.num_posterior_samples is not 0:
            try:
                if not isinstance(importance_resampling,bool):
                    raise DeftError('Q_samples syntax error. Please ensure importance_resampling is of type bool')
            except DeftError as e:
                print(e)
                sys.exit(1)

            # return all samples here
            Q_Samples = []
            sample_weights = []
            for sampleIndex in range(self.num_posterior_samples):
                Q_Samples.append(
                    Density1D(Field1D(self.results_dict['phi_samples'][:, sampleIndex], self.grid, self.bounding_box)))

                sample_weights.append(self.results_dict['phi_weights'][sampleIndex])

            if importance_resampling:

                indices = range(self.num_posterior_samples)
                index_probs = sample_weights / sum(sample_weights)
                weighted_sample_indices = np.random.choice(indices, size=self.num_posterior_samples, p=index_probs)

                Q_samples_weighted = []
                for weight_index in weighted_sample_indices:
                    Q_samples_weighted.append(Q_Samples[weight_index])

                #print("Warning, returning list of Density objects; use index while using evaluate")
                # return weight samples as default
                return Q_samples_weighted

            #print("Warning, returning list of Density objects; use index while using evaluate")
            # we have samples Q_samples in a list

            return Q_Samples

        else:
            print("Q_Samples: Please ensure fit is run and posterior sampling method is not None")

    # returns Q_samples evaluated on the x-values provided
    def eval_samples(self, xs=None):
        """
        Evaluate sampled Qs at specified locations

        parameters:
            xs (np.array): Data points at which samples are evaluated
        return:
            values (np.array): Value of sampled Qs at provided x values. Returns
                None if no samples were taken
        """

        # If xs are not provided use grid
        if xs is None:
            xs = self.get_grid()

        # Evaluate Q_samples on grid and return
        if self.num_posterior_samples > 0:
            Q_samples = self.get_Q_samples()
            values = np.array([Q.evaluate(xs) for Q in Q_samples]).T
        else:
            values = None
        return values

    #
    # The main DEFT algorithm in 1D.
    #

    def run(self):
        """
        Runs DEFT 1D on data. Requires that all relevant input already be set
        as attributes of class instance.

        return:
            results (class instance): A container class whose attributes contain
            the results of the DEFT 1D algorithm.
        """

        # Extract information from Deft1D object
        data = self.data
        G = self.num_grid_points
        alpha = self.alpha
        bbox = self.bounding_box
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
        Q_star_func = lambda x: sp.exp(-phi_star_func(x)) / Z
        results.Q_star_func = Q_star_func

        # Record execution time
        results.copy_compute_time = copy_compute_time
        results.laplacian_compute_time = laplacian_compute_time
        results.deft_1d_compute_time = time.clock()-start_time

        return results