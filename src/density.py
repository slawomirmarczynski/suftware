#!/usr/local/bin/python -W ignore
import scipy as sp
import numpy as np
import sys
import time
from scipy.interpolate import interp1d
import pdb

# Import deft-related code
from src import deft_core
from src import utils
from src import laplacian

from src.utils import DeftError
from src.interpolated_density import InterpolatedDensity
from src.interpolated_field import InterpolatedField

class Density:
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
    get_results_dict():
        Transforms the results object passed by run() into a dictionary, or
        returns a single specified attribute of the results object.
    get_Q_samples():
        Returns a list of sampled Qs in functionalized form.
    eval_samples():
        Evaluates sampled Qs on specified x values.
    """

    def __init__(self,
                 data,
                 alpha=3,
                 grid=None,
                 grid_spacing=None,
                 num_grid_points=None,
                 bounding_box=None,
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

        # Record other inputs as class attributes
        self.alpha = alpha
        self.grid = grid
        self.grid_spacing = grid_spacing
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
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

        # Validate inputs
        self.inputs_check()

        # clean input data
        self.clean_data()

        # Choose grid
        self.set_grid()

        # Fit to data
        self.run()

        # Save some results
        self.results_dict = self.get_results_dict()
        self.histogram = self.results_dict['R']
        self.phi_star = InterpolatedField(self.results_dict['phi_star'],
                                          self.grid,
                                          self.bounding_box)
        self.Q_star = InterpolatedDensity(self.phi_star)
        self.evaluate = self.Q_star.evaluate
        self.values = self.evaluate(self.grid)
        self.sample_values = self.eval_samples(self.grid)


    def set_grid(self):
        """
        Sets the grid based on user input
        :param: self
        :return: None
        """

        data = self.data
        grid = self.grid
        grid_spacing = self.grid_spacing
        num_grid_points = self.num_grid_points
        bounding_box = self.bounding_box
        alpha = self.alpha

        # Compute minimum grid spacing
        data.sort()
        diffs = np.diff(data)
        min_grid_spacing = min(diffs[diffs > 0])

        # If grid is specified
        if grid is not None:
            assert (len(grid) >= 2)
            grid = np.array(grid).copy()
            grid = np.linspace(grid.min(), grid.max(), len(grid))
            num_grid_points = len(grid)
            grid_start = min(grid)
            grid_stop = max(grid)
            grid_spacing = (grid_stop - grid_start)/(num_grid_points-1)
            bounding_box = [grid_start - grid_spacing/2,
                            grid_stop + grid_spacing/2]


        # If grid is not specified
        else:
            # If bounding box is specified
            if bounding_box is not None:
                assert(len(bounding_box) == 2)
                assert(bounding_box[0] < bounding_box[1])
                assert(num_grid_points >= 2*alpha)

            # If bounding box is not specified
            else:
                # Choose bounding box automatically
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

                # Set bounding box
                bounding_box = [lower_bound - (1E-6) * data_span,
                                upper_bound + (1E-6) * data_span]

            # Set bounding box size
            box_size = bounding_box[1] - bounding_box[0]

            # If num_grid_points is spedified
            if (num_grid_points is not None):
                grid_spacing = box_size / num_grid_points

            # If num_grid_points is not specified but grid_spacing is
            elif (grid_spacing is not None):
                num_grid_points = np.floor(box_size/grid_spacing).astype(int)

            # If neither num_grid_points or grid_spacing is specified
            else:
                grid_spacing = max(min_grid_spacing, box_size/100)
                grid_spacing = min(grid_spacing, box_size/(2*alpha))
                num_grid_points = box_size/grid_spacing

            # Set grid
            grid_start = bounding_box[0] + grid_spacing/2
            grid_stop = grid_start + (num_grid_points+1E-6)*grid_spacing
            grid = np.linspace(grid_start, grid_stop, num_grid_points)

            # Set final grid
            self.grid = grid
            self.grid_spacing = grid_spacing
            self.num_grid_points = int(num_grid_points)
            self.bounding_box = bounding_box


    def get_results_dict(self, key=None):
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
                    InterpolatedDensity(InterpolatedField(self.results_dict['phi_samples'][:, sampleIndex], self.grid, self.bounding_box)))

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

        # Store results
        self.results = results

    # Check inputs
    def inputs_check(self):
        """
        Check all inputs NOT having to do with the choice of grid
        :param self:
        :return: None
        """

        # Set variables
        G = self.num_grid_points
        alpha = self.alpha
        bbox = self.bounding_box
        periodic = self.periodic
        Z_eval = self.Z_evaluation_method
        DT_MAX = self.max_t_step
        print_t = self.print_t
        tollerance = self.tolerance
        resolution = self.resolution
        deft_seed = self.seed
        fix_t_at_t_star = self.sample_only_at_l_star
        num_pt_samples = self.num_posterior_samples
        max_log_evidence_ratio_drop = self.max_log_evidence_ratio_drop

        # alpha
        try:
            if not isinstance(alpha, int):
                raise DeftError(
                    'Input check failed. Parameter "alpha" must be an integer: alpha = %s' % type(
                        alpha))
        except DeftError as e:
            print(e)
            sys.exit(1)

        # G (self.num_grid_points)
        if G is not None:
            try:
                if not isinstance(G, int):
                    raise DeftError(
                        'Input check failed. Parameter "num_grid_points" must be an integer: num_grid_points = %s' % type(
                            G))
            except DeftError as e:
                print(e)
                sys.exit(1)

            try:
                if not (G >= 2 * alpha and G <= 1000):
                    raise DeftError(
                        'Input check failed. Parameter "num_grid_points" must between [2*alpha, 1000]: num_grid_points = %s' % G)
            except DeftError as e:
                print(e)
                sys.exit(1)

        try:
            # the following values of alpha are used in the paper
            if not (1 <= alpha <= 4):
                raise DeftError(
                    'Input check failed. Parameter "alpha" must be 1 <= alpha <= 4: alpha = %s' % alpha)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # bbox
        if bbox is not None:
            try:
                bbox = list(bbox)
                assert len(bbox)==2
                bbox[0] = float(bbox[0])
                bbox[1] = float(bbox[1])
                assert bbox[0] < bbox[1]
                bbox = tuple(bbox)

            except TypeError as e:
                print(
                    'Input check failed: bounding_box must be ordered and castable as a 2-tuple of floats. Current bbox: %s' % repr(bbox)
                )
                sys.exit(1)

        # Make sure periodic is valid
        try:
            if not isinstance(periodic, bool):
                raise DeftError(
                    'Input check failed. Parameter "periodic" must be of type boolean: periodic = %s' % type(
                        periodic))
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure Z_eval is valid
        Z_evals = ['Lap', 'Lap+Imp', 'Lap+Fey']
        try:
            if not (Z_eval in Z_evals):
                raise DeftError(
                    'Input check failed. Z_evaluation_method must be in %s: Z_evaluation_method = %s' % (
                    Z_evals, Z_eval))
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure DT_MAX is valid
        try:
            if not isinstance(DT_MAX, utils.NUMBER):
                raise DeftError(
                    'Input check failed. Parameter "max_t_step" must be a number: max_t_step = %s' % type(
                        DT_MAX))
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if not (DT_MAX > 0):
                raise DeftError(
                    'Input check failed. Parameter "max_t_step" must be > 0: max_t_step = %s' % DT_MAX)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure print_t is valid
        try:
            if not isinstance(print_t, bool):
                raise DeftError(
                    'Input check failed. Parameter "print_t" must be a boolean: print_t = %s' % type(
                        print_t))
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure tolerance is valid
        try:
            if not isinstance(tollerance, float):
                raise DeftError(
                    'Input check failed. Parameter "tolerance" must be a float: tolerance = %s' % type(
                        tollerance))
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if not (tollerance > 0):
                raise DeftError(
                    'Input check failed. Parameter "tolerance" must be > 0: tolerance = %s' % tollerance)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure resolution is valid
        try:
            if not isinstance(resolution, utils.NUMBER):
                raise DeftError(
                    'Input check failed. Parameter "resolution" must be a number: resolution = %s' % type(
                        resolution))
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if not (resolution > 0):
                raise DeftError(
                    'Input check failed. Parameter "resolution" must be > 0: resolution = %s' % resolution)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure deft_seed is valid
        try:
            if deft_seed is not None and not isinstance(deft_seed, int):
                raise DeftError(
                    'Input check failed. Parameter "seed" must be either None or an integer: seed = %s' % type(
                        deft_seed))
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if (deft_seed is not None) and (
                (deft_seed < 0) or (deft_seed > 2 ** 32 - 1)):
                raise DeftError(
                    'Input checks failed. Parameter "Seed" must be 0 <= deft_seed <= 2**32-1: deft_seed = %s' % deft_seed)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure fix_t_at_t_star is valid

        try:
            if not isinstance(fix_t_at_t_star, bool):
                raise DeftError(
                    'Input check failed. Parameter "sample_only_at_l_star" must be a boolean: sample_only_at_l_star = %s' % type(
                        fix_t_at_t_star))
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Make sure num_pt_samples is valid
        try:
            if not isinstance(num_pt_samples, int):
                raise DeftError(
                    'Input check failed. Parameter "num_posterior_samples" must be an integer: num_posterior_samples = %s' % type(
                        num_pt_samples))
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if not (num_pt_samples >= 0):
                raise DeftError(
                    'Input check failed. Parameter "num_posterior_samples" must be >= 0: num_posterior_samples = %s' % num_pt_samples)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # ensure that max_log_evidence_ratio is a float
        try:
            if not isinstance(max_log_evidence_ratio_drop, utils.NUMBER):
                raise DeftError(
                    'Input check failed. Parameter "max_log_evidence_ratio" must be a number: tolerance = %s' % type(
                        max_log_evidence_ratio_drop))
        except DeftError as e:
            print(e)
            sys.exit(1)


    # This method will be used to clean user input data; it's for use with the API.
    def clean_data(self):
        """
        Sanitize the assigned data
        :param: self
        :return: None
        """
        data = self.data

        try:
            # if data is a list or set, cast into numpy array
            if type(data) == list or type(data) == set:
                data = np.array(data)
            # if data already np array, do nothing
            elif type(data) == np.ndarray:
                pass
            # think about whether the following is a good idea
            elif type(data) != np.ndarray:
                data = np.array(data)
            else:
                raise DeftError("Error: could not cast data into an np.array")
        except DeftError as e:
            print(e)
            sys.exit(1)

        # remove nan's from the np data array
        data = data[~np.isnan(data)]
        # remove positive or negative infinite values from the np data array
        data = data[~np.isinf(data)]
        # remove complex numbers from data
        data = data[~np.iscomplex(data)]
        # make data floats
        data = data.astype(float)

        try:
            if not (len(data) > 0):
                raise DeftError(
                    'Input check failed, data must have length > 0: data = %s' % data)
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            data_spread = max(data) - min(data)
            if not np.isfinite(data_spread):
                raise DeftError(
                    'Input check failed. Data[max]-Data[min] is not finite: Data spread = %s' % data_spread)
        except DeftError as e:
            print(e)
            sys.exit(1)

        try:
            if not (data_spread > 0):
                raise DeftError(
                    'Input check failed. Data[max]-Data[min] must be > 0: data_spread = %s' % data_spread)
        except DeftError as e:
            print(e)
            sys.exit(1)

        # Set cleaned data
        self.data = data