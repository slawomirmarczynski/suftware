"""
Head module. Contains classes for user interfacing.
"""

# Data simulation
from src.simulate_density_data import run as simulate_density_data
from src.simulate_density_data import VALID_DISTRIBUTIONS \
    as simulate_density_data__distribution_types

# Data examples
from src.example_density_data import run as example_density_data
from src.example_density_data import VALID_DATASETS \
    as example_density_data__datasets

# Density estimation
from src.density import Density
from src.interpolated_density import InterpolatedDensity
from src.interpolated_field import InterpolatedField

from src.utils import DeftError as ControlledError

# Classes that have yet to be written
class Density2D:
    """
    Future class for density estimation in a two dimensional area.
    """
    pass

class DensityJoint:
    """
    Future class for estimating the joint distribution between two
    univariate quantities
    """
    pass

class Survival:
    """
    Future class for computing simple survival curves
    """
    pass

class ProportionalHazards:
    """
    Future class for computing proportional hazards models
    """
    pass

class GeneralizedHazards:
    """
    Future class for computing generalized hazards models
    """
    pass

class IntervalCensoredSurvival:
    """
    Future class for computing interval-censored survival curves
    """
