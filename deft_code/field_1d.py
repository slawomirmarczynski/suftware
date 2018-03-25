#!/usr/local/bin/python -W ignore
import scipy as sp
import numpy as np
import sys
import time
from scipy.interpolate import interp1d
from scipy import interpolate

# Import deft-related code
from deft_code import deft_core
from deft_code import utils
from deft_code import laplacian

from deft_code.supplements import inputs_check
from deft_code.supplements import clean_data
from deft_code.utils import DeftError


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
    def __init__(self, phi, grid, bounding_box):

        # Record input
        self.phi = phi
        self.grid = grid
        self.grid_spacing = grid[1]-grid[0]
        self.bounding_box = bounding_box

        # Interpolate using extended grid and extended phi
        self.evaluate = interpolate.interp1d(self.grid,
                                             self.phi,
                                             kind='cubic',
                                             bounds_error=False,
                                             fill_value='extrapolate',
                                             assume_sorted=True)


