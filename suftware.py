"""
Head module. Contains classes for user interfacing.
"""

from src.simulate_data_1d import run as simulate_data_1d
from src.simulate_data_1d import VALID_DISTRIBUTIONS \
    as simulate_data_1d__distribution_types

from src.example_data_1d import run as example_data_1d
from src.example_data_1d import VALID_DATASETS \
    as example_data_1d__datasets

# Established classes
from src.deft_1d import Density1D
Density1D

from src.interpolated_density_1d import InterpolatedDensity1D
InterpolatedDensity1D

from src.interpolated_field_1d import InterpolatedField1D
InterpolatedField1D

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
