from scipy import interpolate
import scipy as sp
from deft_code import utils


class Field1D:
    """
    This class takes a 1d field object evaluated on a grid and returns
    and instance of the Field1d class containing an interpolated field.
    The interpolation is done via scipy's interpolate package.

    parameters:
    ----------
    phi : (...,N,...) array_like A 1-D array of real values.
    G : (int) number of grid points on which phi is evaluated
    bbox: (int,int) list that specifies interpolation as xmin,xmax

    Attributes:
    ----------
    bin-centers
    interpolated_phi
    xmin
    xmax

    Methods:
    -------
    evaluate(x): returns the interpolant at x. Returns zero if x is
        outside the interpolation range

    Usage:
    -----
    field_1d_Object = Field1D(phi_input,grid_size,bbox)
    field_1d_Object.evaluate(x)
    """

    # constructor: calls the interp1d function which default to
    # linear interpolation
    def __init__(self,phi, G, bbox):

        counts, self.bin_centers = utils.histogram_counts_1d(phi, G, bbox)
        self.interpolated_phi = interpolate.interp1d(self.bin_centers, phi)
        # delete counts since they are not needed in this class
        del counts
        self.xmin = bbox[0]
        self.xmax = bbox[1]

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
    Field1D : (class) instance of class Field1D

    Attributes:
    ----------
    Field1D
    xs
    h
    Z

    Methods:
    -------
    evaluate(x): returns Q(x) at x. Returns zero if x is
        outside the interpolation range

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

    methods
    -------
    fit(data, **kwargs)
    get_params()
    set_params()
    """

    def __init__(self):
        pass

    def fit(self):
        pass

    def get_QStar(self):
        pass

    def get_QSampled(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass

