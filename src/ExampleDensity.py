import numpy as np
from src.Density import Density
from src.GaussianMixtureDensity import GaussianMixtureDensity
from src.utils import check


density_list = []

density_list.append(
    GaussianMixtureDensity(name='GM_Fig1',
                           bounding_box=[-15, 15],
                           weights=[2, 1],
                           mus=[-2, 2],
                           sigmas=[1, 1]))

density_list.append(
    GaussianMixtureDensity(name='Normal',
                           bounding_box=[-5,5],
                           mus=[0],
                           sigmas=[1],
                           weights=[1]))

density_list.append(
    GaussianMixtureDensity(name='GM_Narrow',
                           bounding_box=[-6, 6],
                           weights=[1, 1],
                           mus=[-1.25, 1.25],
                           sigmas=[1, 1]))

density_list.append(
    GaussianMixtureDensity(name='GM_Wide',
                           bounding_box=[-6, 6],
                           weights=[1, 1],
                           mus=[-2, 2],
                           sigmas=[1, 1]))

density_list.append(
    GaussianMixtureDensity(name='GM_Foothills',
                           bounding_box=[-5,12],
                           weights=[1., 1., 1., 1., 1.],
                           mus=[0., 5., 8., 10, 11],
                           sigmas=[2., 1., 0.5, 0.25, 0.125]))

density_list.append(
    GaussianMixtureDensity(name='GM_Accordian',
                           bounding_box=[-5, 13],
                           weights=[16., 8., 4., 2., 1., 0.5],
                           mus=[0., 5., 8., 10, 11, 11.5],
                           sigmas=[2., 1., 0.5, 0.25, 0.125, 0.0625]))

density_list.append(
    GaussianMixtureDensity(name='GM_Goalposts',
                           bounding_box=[-25, 25],
                           weights=[1, 1],
                           mus=[-20, 20],
                           sigmas=[1, 1]))

density_list.append(
    GaussianMixtureDensity(name='GM_Goalposts',
                         bounding_box=[-25, 25],
                         weights=[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                         mus=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
                         sigmas=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))


# Convert list to dictionary
density_names = [d.name for d in density_list]
density_dict = dict(zip(density_names, density_list))
density_names.sort()

def list_example_densities():
    """ Returns a list of valid example density names"""
    return density_names.copy()

def ExampleDensity(name):
    """
    This is actually a function, not a class.
    Returns an pre-defined Density object with the specified name.
    """
    check(name in density_dict.keys(),
          """
Example density name "%s" is invalid. 
Please choose a valid name from:
%s
          """ % (name, '\n'.join(density_names)))

    return density_dict[name]
