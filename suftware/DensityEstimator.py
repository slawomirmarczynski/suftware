import sys
import numbers

import numpy as np
import scipy as sp

from consumedtimetimer import ConsumedTimeTimer

# Import deft-related code
import deft_core
import laplacian
from utils import ControlledError, enable_graphics, check, handle_errors,\
    clean_numerical_input, LISTLIKE
from DensityEvaluator import DensityEvaluator


SMALL_NUM = 1E-6  #? arbitrary, non-coherent with the `utils` module. 
MAX_NUM_GRID_POINTS = 1000
DEFAULT_NUM_GRID_POINTS = 100
MAX_NUM_POSTERIOR_SAMPLES = 1000
MAX_NUM_SAMPLES_FOR_Z = 1000



class DensityEstimator:
    """
    """

    def __init__(self,
                 data,
                 grid=None,
                 step=None,
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
        self.step = step
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

        # Data should be numeric and finite. Thus any NAN/INF values are removed
        # - notice that it is possible that an exception may be raised.
        #
        data = np.mat(data, float)
        data = data[np.isfinite(data)]

        # Choose grid
        self._set_grid()

        # Fit to data
        self._run()

        #? WTF, the results are already avaliable as self.results.*
        #?
        # Save some results 
        #
        self.histogram = self.results.R
        self.maxent = self.results.M
        self.phi_star_values = self.results.phi_star

        # Compute evaluator for density
        #
        self.density_func = DensityEvaluator(self.phi_star_values,
                                             self.grid,
                                             self.bounding_box)

        # Compute optimal density at grid points
        #
        self.values = self.evaluate(self.grid)

        # If any posterior samples were taken
        #
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


        # Define a function for each summary statistic
        def entropy(Q):
            h = self.step
            eps = 1E-10
            assert (all(Q >= 0))
            return np.sum(h * Q * np.log2(Q + eps))

        def mean(Q):
            x = self.grid
            h = self.step
            return np.sum(h * Q * x)

        def variance(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
            return np.sum(h * Q * (x - mu) ** 2)

        def skewness(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
            return np.sum(h * Q * (x - mu) ** 3) / np.sum(
                h * Q * (x - mu) ** 2) ** (3 / 2)

        def kurtosis(Q):
            mu = mean(Q)
            x = self.grid
            h = self.step
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
        h = self.step
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
        step = self.step
        num_grid_points = self.num_grid_points
#?        bounding_box = self.bounding_box
        low = self.low
        high = self.high

        alpha = self.alpha

        # When this method is called there are only two possibilities: 
        # we have got no grid or we have got a grid.
        #
        if grid is None:

            ### Next, define grid based on bounding box ###

            # If step is specified
            if (step is not None):
                assert isinstance(step, float)
                assert np.isfinite(step)
                assert step > 0

                # Set number of grid points
                num_grid_points = np.floor(box_size/step).astype(int)

                # Check num_grid_points isn't too small
                check(2*self.alpha <= num_grid_points,
                      'Using step = %f ' % step +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too small. Reduce step or do not set.')

                # Check num_grid_points isn't too large
                check(num_grid_points <= MAX_NUM_GRID_POINTS,
                      'Using step = %f ' % step +
                      'produces num_grid_points = %d, ' % num_grid_points +
                      'which is too big. Increase step or do not set.')

                # Define grid padding
                # Note: step/2 <= grid_padding < step
                grid_padding = (box_size - (num_grid_points-1)*step)/2
                assert (step/2 <= grid_padding < step)

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
                step = box_size / num_grid_points

                # Define grid padding
                grid_padding = step/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM), # For safety
                                   num_grid_points)

            # Otherwise, set step and num_grid_points based on data
            else:
                assert isinstance(data, np.ndarray)
                assert all(np.isfinite(data))
                assert min(data) < max(data)

                # Compute default grid spacing
                default_step = box_size/DEFAULT_NUM_GRID_POINTS

                # Set minimum number of grid points
                min_num_grid_points = 2 * alpha

                # Set minimum grid spacing
                data.sort()
                diffs = np.diff(data)
                min_step = min(diffs[diffs > 0])
                min_step = min(min_step,
                                       box_size/min_num_grid_points)

                # Set step
                step = max(min_step, default_step)

                # Set number of grid points
                num_grid_points = np.floor(box_size/step).astype(int)

                # Set grid padding
                grid_padding = step/2

                # Define grid to be centered in bounding box
                grid_start = lower_bound + grid_padding
                grid_stop = upper_bound - grid_padding
                grid = np.linspace(grid_start,
                                   grid_stop * (1 + SMALL_NUM),  # For safety
                                   num_grid_points)


        if grid is not None:

            # Check and set number of grid points
            num_grid_points = len(grid)
            assert(num_grid_points >= 2*alpha)

            # Check and set grid spacing
            diffs = np.diff(grid)
            step = diffs.mean()
            assert (step > 0)
            assert (all(np.isclose(diffs, step)))

            # Check and set grid bounds
            grid_padding = step / 2
            lower_bound = grid[0] - grid_padding
            upper_bound = grid[-1] + grid_padding
            bounding_box = np.array([lower_bound, upper_bound])
            box_size = upper_bound - lower_bound


        # Set final grid
        self.grid = grid
        self.step = step
        self.grid_padding = grid_padding
        self.num_grid_points = num_grid_points
        self.bounding_box = bounding_box
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.box_size = box_size


        # Set bin edges
        self.bin_edges = np.concatenate(([lower_bound],
                                         grid[:-1]+step/2,
                                         [upper_bound]))

    def _init_bbox(self):

        #? Assuming that data is a matrix n-by-2 with x, y values

        PADDING_FACTOR = 0.2

        x1 = np.min(data[:, 0])
        x2 = np.max(data[:, 0])
        self.low = x1 - (x2 - x1) * PADDING_FACTOR
        self.high = x2 + (x2 - x1) * PADDING_FACTOR

        if self.low < 0 and x1 >= 0:
            self.low = 0

        if self.high > 0 and x2 > 0:
            self.high = 0

