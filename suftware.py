"""This module contains the classes Deft1D, Field1D, and Density1D See doc-strings for each class for details"""

from scipy import interpolate
import numpy as np
import scipy as sp
from deft_code import utils
from deft_code import deft_1d
from deft_code.supplements import inputs_check
from deft_code.supplements import clean_data
from deft_code.utils import DeftError
import sys


class Deft1D:
    """This class will serve as the interface for running
    deft1d

    parameters
    ----------
    data: (array like)
        User input data for which Deft1D will estimate the density.
    num_grid_points: (int)
        Number of grid points.
    alpha: (int)
        Smoothness parameter. Represents the order of the
        derivative in the action.
    bounding_box: ((int,int) or [int,int])
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
    resolution: (positive float)
        Specifies max distance between neighboring points on the MAP curve.
    seed: (int)
        Specify random seed for evaluation of the partition function
        and for the posterior sampling.
    max_t_step: (non-negative float)
        Maximum t step size on the MAP curve.
    max_log_evidence_ratio_drop: (float)
        Stop criterion for traversing the MAP curve; deft stops when:
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop
    tolerance: (positive float)
        Value which species convergence of phi.
    num_posterior_samples: (non-negative int)
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
    fit(data, **kwargs):
        Fit the data using deft for density estimation.
    get_params():
        Returns the parameters used in the constructor.
    set_params():
        Set parameters for the constructor.
    get_h():
        Returns grid bin width.
    get_bounding_box():
        Return bounding box.
    get_grid():
        Returns the grid as a numpy array.
    get_num_grid_points
        Returns the number of grid points.
    get_results():
        Returns the results object.
    get_phi_star():
        Returns the Field1D object.
    get_Q_star():
        Returns Qstar as a Density1D object.
    get_Q_samples():
        Returns posterior samples of the density.
        keyword arguments:
            get_sample_number: (non-negative int)
                returns the posterior sample specified by argument
            get_first_n_samples (non negative int)
                return the first n samples specified by argument
    """

    def __init__(self,
                 data,
                 num_grid_points=100,
                 alpha=3,
                 bounding_box='Auto',
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
        inputs_check(self)

        # clean input data
        self.data, self.min_h = clean_data(data)

        if self.bounding_box == 'Auto':


            data_spread = np.max(self.data) - np.min(self.data)
            #bbox_left = int(np.min(self.data) - 0.2 * data_spread)
            #bbox_right = int(np.max(self.data) + 0.2 * data_spread)
            bbox_left = round(np.min(self.data) - 0.2 * data_spread,2)
            bbox_right = round(np.max(self.data) + 0.2 * data_spread,2)
            self.bounding_box = [bbox_left, bbox_right]

            #print('bbox_right: ', bbox_right)
            #print('bbox: ',self.bounding_box)

        # make sure G (and therefore step-size is appropriate set based on data).
        elif self.bounding_box != 'Auto':
            if (self.num_grid_points != int((self.bounding_box[1] - self.bounding_box[0]) / self.min_h) and int(
                        (self.bounding_box[1] - self.bounding_box[0]) / self.min_h) <= 1000):
                self.num_grid_points = int((self.bounding_box[1] - self.bounding_box[0]) / self.min_h)
                print(
                'Warning, updating value of num_grid_points based on bounding box entered: ', self.num_grid_points)

        # Fit to data
        self.results = deft_1d.run(self)

        #print('Deft1D ran successfully')
        self.grid = self.get_grid()
        self.results_dict = self.get_results()
        self.histogram = self.results_dict['R']
        self.grid_spacing = self.get_h()
        self.values = self.eval(self.grid)
        self.sample_values = self.eval_samples(self.grid)


    def get_results(self, key=None):
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

    # get step size
    def get_h(self):
        counts, bin_centers = utils.histogram_counts_1d(self.results.__dict__.get('phi_star'), self.num_grid_points, self.bounding_box)
        del counts
        # h = bc[1]-bc[0]
        return bin_centers[1]-bin_centers[0]

    # return bounding box
    def get_bounding_box(self):
        return self.bounding_box

    # return number of grid points
    def get_num_grid_points(self):
        return self.num_grid_points

    # return xs of grid
    def get_grid(self):
        counts, bin_centers = utils.histogram_counts_1d(self.results.__dict__.get('phi_star'), self.num_grid_points, self.bounding_box)
        del counts
        # h = bc[1]-bc[0]
        return bin_centers

    # returns a Field1D object
    def get_phi_star(self):

        if self.results is not None:
            return Field1D(self.results.__dict__.get('phi_star'), self.num_grid_points, self.bounding_box)
        else:
            print("phi is none. Please run fit first.")

    # returns a Density1D object
    def get_Q_star(self):

        if self.results is not None:
            return Density1D(self.get_phi_star())
        else:
            print("Q_star is none. Please run fit first.")

    # returns Q_star evaluated on the x-values provided
    def eval(self, xs=None):

        # If xs are not provided use grid
        if xs is None:
            xs = self.get_grid()

        # Evaluate Q_star on grid and return
        Q_star = self.get_Q_star()
        return Q_star.evaluate(xs)

    # Returns the histogram R
    def get_R(self):
        results_dict = self.get_results()
        return results_dict['R']

    # if importance_resampling == True:
    #   then
    def get_Q_samples(self, importance_resampling=True):

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
                    Density1D(Field1D(self.get_results()['phi_samples'][:, sampleIndex], self.num_grid_points, self.bounding_box)))

                sample_weights.append(self.get_results()['phi_weights'][sampleIndex])

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

    # returns Q_star evaluated on the x-values provided
    def eval_samples(self, xs=None):

        # If xs are not provided use grid
        if xs is None:
            xs = self.get_grid()

        # Evaluate Q_samples on grid and return
        if self.num_posterior_samples > 0:
            Q_samples = self.get_Q_samples()
            return np.array([Q.evaluate(xs) for Q in Q_samples]).T
        else:
            return None


    def get_params(self,key=None):
        if key is None:
            # if no key provided, return all parameters
            return self.__dict__
        else:
            try:

                if '__getattribute__' in dir(self):
                    # recommended way of retrieving attributes
                    return self.__getattribute__(key)
                else:
                    # attribute retrieval for python 2 and older
                    return self.__dict__[key]

            except (AttributeError, KeyError) as e:
                print("Get Params: parameter does not exist: ",e)

    # should check if parameter exists in __dict__
    def set_params(self,parameter=None,value=None, **kwargs):
        # if no dictionary provided
        if bool(kwargs) is False:

            # check validity of parameter key
            try:
                if not parameter in self.__dict__:
                    raise DeftError('Error in set params: setting invalid parameter, '+parameter)

                elif '__setattr__' in dir(self):
                    self.__setattr__(parameter, value)
                else:
                    self.__dict__[parameter] = value

            except DeftError as e:
                print(e)
                #sys.exit(1)

        else:

            # check validity of parameter dictionary
            try:
                for key in kwargs:

                    if not key in self.__dict__.keys():
                        raise DeftError('Error in set params: setting invalid parameter, ' + key)

                    elif '__setattr__' in dir(self):
                        self.__setattr__(key, kwargs[key])
                    else:
                        self.__dict__[key] = kwargs[key]
            except DeftError as e:
                print(e)
                #sys.exit(1)




class Field1D:
    """
    This class takes a 1d field object evaluated on a grid and returns
    and instance of the Field1d class containing an interpolated field.
    The interpolation is done via scipy's interpolate package.

    parameters:
    ----------
    phi: (...,N,...)
        array_like A 1-D array of real values.
    G: (int)
        number of grid points on which phi is evaluated
    bbox: ([int,int])
        list that specifies interpolation as xmin, xmax

    Attributes:
    ----------
    bin-centers: (array like)
        corresponds to the x values where phi is evaluated
    interpolated_phi: (object)
        the interpolated field

    Methods:
    -------
    evaluate(x):
        returns the interpolant at x. Returns zero if x is
        outside the interpolation range

    Usage:
    -----
    field_1d_Object = Field1D(phi_input,grid_size,bbox)
    field_1d_Object.evaluate(x)
    """

    # constructor: calls the interp1d function which default to
    def __init__(self,phi, G, bbox):

        counts, self.bin_centers = utils.histogram_counts_1d(phi, G, bbox)
        self.interpolated_phi = interpolate.interp1d(self.bin_centers, phi)
        # delete counts since they are not needed in this class
        del counts

    # method to evaluate the interpolant at a given x value
    def evaluate(self,x):
        # returns the interpolant at x
        try:
            return self.interpolated_phi.__call__(x)
        # handle value
        except ValueError:
            print(" The x value is out of the interpolation range.")
            return sp.nan


class Density1D:
    """
    This class forms a Density object (Q) based on the
    interpolated field object from the Field1D class

    parameters:
    ----------
    Field1D: (class)
        instance of class Field1D

    Attributes:
    ----------
    Field1D: (class)
        this field attribute is an input to the constructor
        during initialization of the Density1D instance. It's an
        attribute so that it may be used in the evaluate function
        for all instances of Density1D.
    xs: (array like)
        corresponds to the x values where Q (or phi) is evaluated
    h:  (float)
        bin width
    Z:  (float)
        partition/normalization function

    Methods:
    -------
    evaluate(x):
        returns Q(x) at x. Returns zero if x is outside the
        interpolation range

    Usage:
    -----
    density1DObject = Density1D(fieldObject)
    density1DObject.evaluate(x)
    """

    def __init__(self, Field1D):
        self.Field1D = Field1D
        self.xs = self.Field1D.bin_centers
        self.h = self.xs[1]-self.xs[0]
        self.Z = sp.sum(self.h * sp.exp(-self.Field1D.evaluate(self.xs)))

    def evaluate(self,x):
        try:
            # return Q(x)
            return sp.exp(-self.Field1D.evaluate(x)) / self.Z
        except:
            print("Error: x value out of interpolation range")
            return sp.nan