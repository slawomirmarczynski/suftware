"""
Head module. Contains classes for user interfacing.
"""

# Make classes local
from src.DensityEstimator import DensityEstimator
from src.SimulatedDataset import SimulatedDataset
from src.ExampleDataset import ExampleDataset
from src.Density import Density
from src.GaussianMixtureDensity import GaussianMixtureDensity
from src.utils import ControlledError as ControlledError
from src.ExampleDensity import ExampleDensity, list_example_densities
from src.ScipyStatsDensity import ScipyStatsDensity

# Enable plotting
from src.utils import enable_graphics, check, ControlledError

# # Classes that have yet to be written
# class Density2DEstimator:
#     """
#     Future class for density estimation in a two dimensional area.
#     """
#     pass
#
# class JointDensityEstimator:
#     """
#     Future class for estimating the joint distribution between two
#     univariate quantities.
#     """
#     pass
#
# class SurvivalCurveEstimator:
#     """
#     Future class for computing simple survival curves
#     """
#     pass
#
# class ProportionalHazardsEstimator:
#     """
#     Future class for computing proportional hazards models
#     """
#     pass
#
# class GeneralizedHazardsEstimator:
#     """
#     Future class for computing generalized hazards models
#     """
#     pass
#
# class IntervalCensoredSurvivalEstimator:
#     """
#     Future class for computing interval-censored survival curves
#     """
#

# try_me functions
def demo(example='real_data'):
    """
    Performs a demonstration of suftware.

    Parameters
    ----------

    example: (str)
        A string specifying which demo to run. Must be 'real_data',
        'simulated_data', or 'custom_data'.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)

    example_dict = {
        'custom_data': 'docs/example_custom.py',
        'simulated_data': 'docs/example_wide.py',
        'real_data': 'docs/example_alcohol.py'
    }

    check(example in example_dict,
          'example = %s is not valid. Must be one of %s'%\
          (example, example_dict.keys()))

    file_name = '%s/%s'%(example_dir, example_dict[example])
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s'%\
              (file_name, line, content, line))
    exec(open(file_name).read())
