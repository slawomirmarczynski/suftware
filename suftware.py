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

# Promoter classes to direct membership in the suftware module
from deft_code.deft_1d import Deft1D, Density1D, Field1D
