"""This module contains the classes Deft1D, Field1D, and Density1D See doc-strings for each class for details"""


# Promote classes to direct membership in the suftware module
from src.deft_1d import Deft1D
from src.density_1d import Density1D
from src.field_1d import Field1D

from src.simulate_data_1d import run as simulate_data_1d
from src.simulate_data_1d import VALID_DISTRIBUTIONS \
    as simulate_data_1d__distribution_types

from src.example_data_1d import run as example_data_1d
from src.example_data_1d import VALID_DATASETS \
    as example_data_1d__datasets
