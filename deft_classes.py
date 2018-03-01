from scipy import interpolate
import numpy as np
import scipy as sp
from deft_code import utils
from deft_code import deft_1d
from deft_code.supplements import inputs_check
from deft_code.supplements import clean_data
from deft_code.utils import DeftError
import sys

"""
This module contains the classes Deft1D, Field1D, and Density1D
See doc-strings for each class for details
"""


class Deft1D:
    """This class will serve as the interface for running
    deft1d

    parameters
    ----------
    data: array like
        User input data for which Deft1D will estimate the density
    num_grid_points: (int) number of grid points
    alpha: (int)
        smoothness parameter. Represents the order of the
        derivative in the action
    bounding_box: ([int,int])
        bounding box for density estimation
    periodic: (boolean)
        Enforce periodic boundary conditions via True or False
    Z_evaluation_method: (string)
        method of evaluation of partition function. Possible Z_eval values:
        'Lap'         : Laplace approximation (default)
        'Lap+Imp' : Laplace approximation + importance sampling
        'Lap+Fey'     : Laplace approximation + Feynman diagrams
    num_Z_samples: (int) -> 'num_samples_for_Z'
        *** Note *** this parameter works only when Z_eval is 'Lap+Sam'
        number of samples for the evaluation of the partition function.
        More samples will help to evaluate a better Q*. More samples
        will also make calculation slower. 0 means the laplace approximation
        for the evaluation of the partition function is used.
    resolution (float):
        Specifies max distance between neighboring points on the MAP curve
    seed: (int)
        specify random seed for posterior sampling methods
    max_t_step: (float) ->
        maximum dt step size on the map curve
    max_log_evidence_ratio_drop: (float)
        stop criterion for traversing the map curve; deft stops when i.e.:
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop
    tolerance: (float)
        Value which species convergence of phi
    posterior_sampling_method: (string) Methods of posterior sampling. Possible values:
         None       : no sampling will be performed (default)
        'Lap'   : ....
        'Lap+Imp'   : sampling from Laplace approximation + importance weight
    num_posterior_samples: (int)
        number of posterior samples.

    sample_only_at_l_star: (boolean)
        if True, than posterior samples drawn at t_star
        if False, posterior sampling done among t near t_star

    attributes
    ----------
    *** Note ***: all parameters are attributes except for results.
    results: (dict)
        contains the results of the deft fit.

    methods
    -------
    fit(data, **kwargs):
        calls the run method in the deft_1d module
    get_params():
        returns the parameters used in the Deft1D constructor
    set_params():
        set parameters for the constructor
    get_h():
        returns grid step size
    get_bounding_box():
        return bounding box
    get_grid():
        returns the grid as a numpy array
    get_num_grid_points
        returns the number of grid points
    get_Results():
        returns the results object
    get_phi():
        returns the Field1D object
    get_Qstar():
        returns Qstar as a Density1D object
    get_Qsampled():
        returns posterior samples of the density
    """

    def __init__(self, data, num_grid_points=100, alpha=3, bounding_box=None, periodic=False, Z_evaluation_method='Lap', num_Z_samples=0, max_t_step=1.0,
                 print_t=False, tolerance=1E-6, resolution=0.1, seed=None, posterior_sampling_method='Lap+W', num_posterior_samples=5, sample_only_at_l_star=False):

        # set class attributes
        self.data = data
        self.G = num_grid_points
        self.alpha = alpha
        self.bbox = bounding_box
        self.periodic = periodic
        self.Z_eval = Z_evaluation_method
        self.num_Z_samples = num_Z_samples
        self.DT_MAX = max_t_step
        self.print_t = print_t
        self.tolerance = tolerance
        self.deft_seed = seed
        self.resolution = resolution
        self.pt_method = posterior_sampling_method
        self.num_pt_samples = num_posterior_samples
        self.fix_t_at_t_star = sample_only_at_l_star
        self.results = None

        # clean input data
        self.data = clean_data(data)

        # set reasonable bounding box based on data, unless user provides bbox.
        # Could be changed.
        if self.bbox is None:
            data_spread = np.max(self.data) - np.min(self.data)
            bbox_left = int(np.min(self.data) - 0.2 * data_spread)
            bbox_right = int(np.max(self.data) + 0.2 * data_spread)
            self.bbox = [bbox_left, bbox_right]

        # Check inputs
        inputs_check(G=self.G, alpha=self.alpha, bbox=self.bbox,
                     periodic=self.periodic, Z_eval=self.Z_eval, DT_MAX=self.DT_MAX,
                     print_t=self.print_t, tollerance=self.tolerance, resolution=self.resolution,
                     deft_seed=self.deft_seed, pt_method=self.pt_method,
                     fix_t_at_t_star=self.fix_t_at_t_star, num_pt_samples=self.num_pt_samples)

    def fit(self):

        # Run deft_1d
        try:
            # ensure that number of posterior samples aren't zero when
            # pt_method is 'Lap+W', 'Lap', or 'MMC'
            if (self.pt_method == 'Lap+W' or self.pt_method == 'Lap') and self.num_pt_samples is 0:
                #self.num_pt_samples = 1000
                self.num_pt_samples = 0

            self.results = deft_1d.run(data=self.data, G=self.G, alpha=self.alpha, bbox=self.bbox,
                                       periodic=self.periodic, Z_eval=self.Z_eval, num_Z_samples=self.num_Z_samples,
                                       DT_MAX=self.DT_MAX, print_t=self.print_t, tollerance=self.tolerance,
                                       resolution=self.resolution, deft_seed=self.deft_seed, pt_method=self.pt_method,
                                       num_pt_samples=self.num_pt_samples, fix_t_at_t_star=self.fix_t_at_t_star)

            print('Deft1D ran successfully')
            return self.results

        # this should be more specific
        except:
            # include include message with more details here
            # *** Ask what to do ***
            print('Deft fit failed')

    def get_results(self, key=None):
        if self.results is not None and key is None:
            # return the dictionary containing results if no key provided
            return self.results.__dict__
        elif self.results is not None and key is not None:
            try:
                #return self.results.__dict__.get(key)
                return self.results.__getattribute__(key)
            except AttributeError as e:
                print("Get results:",e)
        else:
            print("Get Results: Deft results are none. Please run fit first.")

    # get step size
    def get_h(self):
        counts, bin_centers = utils.histogram_counts_1d(self.results.__dict__.get('phi_star'), self.G, self.bbox)
        del counts
        # h = bc[1]-bc[0]
        return bin_centers[1]-bin_centers[0]

    # return bounding box
    def get_bounding_box(self):
        return self.bbox

    # return number of grid points
    def get_num_grid_points(self):
        return self.G

    # return xs of grid
    def get_grid(self):
        counts, bin_centers = utils.histogram_counts_1d(self.results.__dict__.get('phi_star'), self.G, self.bbox)
        del counts
        # h = bc[1]-bc[0]
        return bin_centers

    # returns a Field1D object
    def get_phi(self):

        if self.results is not None:
            return Field1D(self.results.__dict__.get('phi_star'), self.G, self.bbox)
        else:
            print("phi is none. Please run fit first.")

    # returns a Density1D object
    def get_Qstar(self):

        if self.results is not None:
            return Density1D(self.get_phi())
        else:
            print("Q_star is none. Please run fit first.")

    def get_Qsampled(self,get_sample_number=None, get_first_n_samples=None):

        # ensure parameters are legal
        if self.results is not None and self.num_pt_samples is not 0:
            try:
                if not isinstance(get_sample_number,int) and get_sample_number is not None:
                    raise DeftError('Q_sample syntax error. Please ensure get_sample_number is of type int')
            except DeftError as e:
                print(e)
                sys.exit(1)

            try:
                if not isinstance(get_first_n_samples,int) and get_first_n_samples is not None:
                    raise DeftError('Q_sample syntax error. Please ensure get_first_n_samples is of type int')
            except DeftError as e:
                print(e)
                sys.exit(1)

            if get_sample_number is not None and get_first_n_samples is not None:
                print("Q_sample Warning: both parameters (get_sample_number, get_first_n_samples) used, please use only one parameter.")

            # return a single sample chosen by the user
            if get_sample_number is not None:

                if get_sample_number >= 0 and get_sample_number < self.num_pt_samples:
                    # return Q_sample specified by the user.
                    return Density1D(Field1D(deft.get_results()['phi_samples'][:, get_sample_number],self.G,self.bbox))
                elif get_sample_number < 0:
                    print("Q_sample error: Please set get_sample_number >= 0, exiting...")
                    # need to exit in this case because evaluate will throw an error.
                    sys.exit()
                elif get_sample_number >= self.num_pt_samples:
                    print('Q_sample error: Please ensure get_sample_number < number of posterior samples, exiting...')
                    # need to exit in this case because evaluate will throw an error.
                    sys.exit()

            # get first n samples. This method could be modified to return a range of samples
            elif get_first_n_samples is not None:

                if get_first_n_samples < 0:
                    print("Q_sample: please set 'get_first_n_samples' > 0")
                    sys.exit()

                elif get_first_n_samples > self.num_pt_samples:
                    print("Q_sample: please set 'get_first_n_samples' < number of posterior samples")
                    sys.exit()

                elif get_first_n_samples >= 0:
                    # list containing samples
                    Q_Samples = []
                    for sampleIndex in range(get_first_n_samples):
                        Q_Samples.append(Density1D(Field1D(deft.get_results()['phi_samples'][:, sampleIndex],self.G,self.bbox)))
                    print("Warning, returning list of Density objects; use index while using evaluate")
                    return Q_Samples

            # get all samples
            else:
                # return all samples here
                Q_Samples = []
                for sampleIndex in range(self.num_pt_samples):
                    Q_Samples.append(
                        Density1D(Field1D(deft.get_results()['phi_samples'][:, sampleIndex], self.G, self.bbox)))
                print("Warning, returning list of Density objects; use index while using evaluate")
                return Q_Samples

        else:
            print("Q_Samples: Please ensure fit is run and posterior sampling method is not None")

    def get_params(self,key=None):
        if key is None:
            # if no key provided, return all parameters
            return self.__dict__
        else:
            try:
                return self.__getattribute__(key)
            except AttributeError as e:
                print("Get Params:",e)

    # should check if parameter exists in __dict__
    def set_params(self,parameter=None,value=None, **kwargs):
        # if no dictionary provided
        if bool(kwargs) is False:
            self.__setattr__(parameter, value)
        else:
            for key in kwargs:
                self.__setattr__(key, kwargs[key])


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


########################################################
########################################################
# WARNING THE FOLLOWING IS TEMPORARY AND WILL BE DELETED
########################################################
########################################################


import matplotlib.pyplot as plt

# Use cases/tests

#print(Deft1D.__doc__)

# load data
#data = np.loadtxt('./data/old_faithful_eruption_times.dat').astype(np.float)
#data = np.loadtxt('./data/old_faithful_eruption_times.dat')
data = np.genfromtxt('./data/old_faithful_eruption_times.dat')

# initialize deft object
deft = Deft1D(data)
# run fit
deft.fit()


#Qstar = deft.get_Qstar()
#print(Qstar.evaluate(0.1))
#print(deft.get_results()['Q_star'])

Q_samples = deft.get_Qsampled()
#print(Q_samples)
#Q_samples = deft.get_Qsampled(get_sample_number=-1)
#print(Q_samples)
#print(Q_sample.evaluate(0.5))


# to get column x/ sample x, do deft.get_results()['phi_samples'][:,1]
#print(deft.get_results()['phi_samples'][:,:2])
# to get first n samples samples deft.get_results()['phi_samples'][:,:n], print warning if n> num_pt_samples
#print(deft.get_results()['phi_samples'][:,:3])

# to get all samples
#print(deft.get_results()['phi_samples'])
#Qstar = deft.get_Qstar()


#Q_samples = deft.get_Qsampled()
#print(Q_samples.evaluate(0.1))
#print(Qstar)
# check why 0.01 and 0.02 is failing
#print(Qstar.evaluate(0.5))
#Qstar.evaluate(0.34)
xs = deft.get_grid()


#plt.plot(xs,Qstar.evaluate(xs),'o')
#plt.show()

#print(deft.get_h())
#print(deft.get_bounding_box())

#print(deft.get_grid())

#print(deft.get_results())

# get one parameter value
#print(deft.get_params('G'))

# get all parameters
#print(deft.get_params())

# get parameter by key
#deft.get_params('resolution')

# get Q_star
#deft.get_Qstar()

# get Q_samples
#deft.get_Qsampled()

# get deft results
#deft.get_results()

# get deft result by key
#deft.get_results('phi_star')

# access particular results pythonically
#print(deft.get_results()['phi_star'])

# set individual parameters by key, value
#deft.set_params('G',10)

# set parameters via dictionary
#d = {"G":10,"alpha":2}
#deft.set_params(**d)
#print(deft.get_Qsampled())


#field = Field1D(deft.get_results()['phi_star'], deft.get_params('G'), deft.get_params('bbox'))
#print(field.evaluate(0.75))

#density = Density1D(field)
#print(density.evaluate(0.5))

#print(density.evaluate(0.5))

# should result density object
#Q_star = deft.get_Qstar()
#print(Q_star)

#print(deft.get_results()['phi_star'])
#print(deft.get_params('alpha'))
#print(deft.get_params())
#deft.set_params('G',10)
#print(deft.get_params())
