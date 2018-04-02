"""
Head module. Contains classes for user interfacing.
"""

# DensityEstimator estimation
from src.DensityEstimator import DensityEstimator

# Data simulation
from src.SimulatedData import SimulatedData

# Data examples
from src.ExampleData import ExampleData

# Interpolators
from src.DensityEvaluator import DensityEvaluator
from src.interpolated_field import InterpolatedField

# Error handling
from src.utils import ControlledError as ControlledError

# Enable plotting
from src.utils import enable_graphics

# Classes that have yet to be written
class Density2DEstimator:
    """
    Future class for density estimation in a two dimensional area.
    """
    pass

class JointDensityEstimator:
    """
    Future class for estimating the joint distribution between two
    univariate quantities.
    """
    pass

class SurvivalCurveEstimator:
    """
    Future class for computing simple survival curves
    """
    pass

class ProportionalHazardsEstimator:
    """
    Future class for computing proportional hazards models
    """
    pass

class GeneralizedHazardsEstimator:
    """
    Future class for computing generalized hazards models
    """
    pass

class IntervalCensoredSurvivalEstimator:
    """
    Future class for computing interval-censored survival curves
    """

