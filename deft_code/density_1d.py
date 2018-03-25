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
from deft_code.field_1d import Field1D

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

    def __init__(self, field):
        self.field = field
        self.bounding_box = self.field.bounding_box
        self.Z = sp.sum(self.field.grid_spacing * sp.exp(-self.field.phi))

    def evaluate(self, x, outside_bbox=0):

        # Convert to numpy array
        x = np.array(x)

        # If dimension is zero, put in numpy array and rerun
        if len(x.shape)==0:
            array = self.evaluate(np.array([x]),
                                  outside_bbox=outside_bbox)
            return array[0]

        # Otherwise, evaluate
        else:
            values = sp.exp(-self.field.evaluate(x)) / self.Z
            if outside_bbox != 'interp':
                indices = (x < self.bounding_box[0]) | (x > self.bounding_box[1])
                values[indices] = outside_bbox
            return values
