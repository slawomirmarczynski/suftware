import numpy as np
import scipy.stats as ss
from src.utils import grid_info_from_bbox_and_G
from src.Density import Density
from src.utils import check

class ScipyStatsDensity(Density):

    def __init__(self, ss_obj, bounding_box=None, num_gridpoints=10000,
                 **ss_obj_params):

        # If bounding box is not specified, created one containing central
        # 99% of data
        if bounding_box is None:
            xmin = ss_obj.ppf(0.005, **ss_obj_params)
            xmax = ss_obj.ppf(0.995, **ss_obj_params)
            bounding_box = [xmin, xmax]

        # Compute grid
        h, grid, bin_edges = grid_info_from_bbox_and_G(bbox=bounding_box,
                                                       G=num_gridpoints)

        # Compute values on grid
        values = ss_obj.pdf(grid, **ss_obj_params)
        values /= sum(h*values)

        # Call density constructor
        Density.__init__(self,
                         grid=grid,
                         values=values,
                         interpolation_method='cubic',
                         min_value=1E-20)

        # Save ss_obj and parameters
        self.ss_obj = ss_obj
        self.ss_obj_params = ss_obj_params
