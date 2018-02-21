from scipy import interpolate
import scipy as sp
from deft_code import utils
from deft_code import deft_1d


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
        # the 0.1's are heuristic numbers that seem to work well with interp1d
        self.xs = self.Field1D.bin_centers
        self.h = self.xs[1]-self.xs[0]
        self.Z = sp.sum(self.h * sp.exp(-self.Field1D.evaluate(self.xs)))

    def evaluate(self,x):
        try:
            # return Q(x)
            return sp.exp(-self.Field1D.evaluate(x)) / self.Z
        except:
            print("Error: please x value out of interpolation range")


class Deft1D:
    """This class will serve as the interface for running
    deft1d

    parameters
    data: array like
        User input data for which Deft1D will estimate the density
    G: (int) number of grid points
    alpha: (int)
        smoothness parameter. Represents the order of the
        derivative in the action
    bbox: ([int,int])
        bounding box for density estimation
    periodic: (boolean)
        Enforce periodic boundary conditions via True or False
    Z_eval: (string)
        method of evaluation of partition function. Possible Z_eval values:
        'Lap'         : Laplace approximation (default)
        'Lap+Sam[+P]' : Laplace approximation + importance sampling
        'GLap[+P]'    : generalized Laplace approximation
        'GLap+Sam[+P]': generalized Laplace approximation + importance sampling
        'Lap+Fey'     : Laplace approximation + Feynman diagrams
        Note: [+P] means this task can be done in parallel
    num_Z_samples: (int)
        *** Note *** this parameter works only when Z_eval is 'Lap+Sam'
        number of samples for the evaluation of the partition function.
        More samples will help to evaluate a better Q*. More samples
        will also make calculation slower. 0 means the laplace approximation
        for the evaluation of the partition function is used.
    resolution (float):
            Specifies max distance between neighboring points on the
            MAP curve
    DT_MAX: (float)
        maximum dt step size on the map curve
    tolerance: (float)
        Value which species convergence of phi
    pt_method: (string) Methods of posterior sampling. Possible values:
         None       : no sampling will be performed (default)
        'Lap[+P]'   : sampling from Laplace approximation + importance weight
        'GLap[+P]'  : sampling from generalized Laplace approximation + importance weight
        'MMC'       : sampling using Metropolis Monte Carlo
        Note: [+P] means this task can be done in parallel
    num_pt_samples: (int)
        number of posterior samples.
    fix_t_at_t_star: (boolean)
        if True, than posterior samples drawn at t_star
        if False, posterior sampling done among t near t_star

    methods
    -------
    fit(data, **kwargs):
        calls the run method in the deft_1d module
    get_params():
        returns the parameters used in the Deft1D constructor
    set_params():
        set parameters for the constructor
    get_Results():
        returns the results object
    get_Qstar():
        returns the Qstar attribute from results
    get_Qsampled(): ?
    """

    def __init__(self, data, G=100, alpha=3, bbox=[-6,6], periodic=False, Z_eval='Lap', num_Z_samples=0, DT_MAX=1.0,
                 print_t=False, tolerance=1E-6, resolution=0.1, pt_method=None, num_pt_samples=0,fix_t_at_t_star=False):

        # set class attributes
        self.data = data
        self.G = G
        self.alpha = alpha
        self.bbox = bbox
        self.periodic = periodic
        self.Z_eval = Z_eval
        self.num_Z_samples = num_Z_samples
        self.DT_MAX = DT_MAX
        self.print_t = print_t
        self.tolerance = tolerance
        self.resolution = resolution
        self.pt_method = pt_method
        self.num_pt_samples = num_pt_samples
        self.fix_t_at_t_star = fix_t_at_t_star

        self.outcome_good = False

    def fit(self):

        # Run deft_1d
        try:
            self.results = deft_1d.run(data=self.data, G=self.G, alpha=self.alpha, bbox=self.bbox,
                                       periodic=self.periodic, Z_eval=self.Z_eval, num_Z_samples=self.num_Z_samples,
                                       DT_MAX=self.DT_MAX, print_t=self.print_t, tollerance=self.tollerance,
                                       resolution=self.resolution, deft_seed=self.deft_seed, pt_method=self.pt_method,
                                       num_pt_samples=self.num_pt_samples, fix_t_at_t_star=self.fix_t_at_t_star)
            print('Succeeded!  t_star = %.2f' % self.results.t_star)
            print(self.pt_method)

        except:
            # include include message with more details here
            print('Deft fit failed')

    def get_Results(self):
        return self.results

    def get_Qtar(self):
        # double get this
        return self.results.results.Q_star

    def get_Qsampled(self):
        pass

    def get_params(self):
        # return constructor parameters
        pass

    def set_params(self):
        pass


print(Field1D.__doc__)