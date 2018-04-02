import numpy as np
from scipy import interpolate
from src.utils import check

class DensityEvaluator:
    """
    A probability density that can be evaluated at anywhere

    Parameters
    ----------

    field_values: (1D np.array)

        The values of the field used to computed this density.

    grid: (1D np.array)

        The grid points at which the field values are defined. Must be the same
        the same shape as field.

    Attributes
    ----------

    field_values:
        See above.

    grid:
        See above.

    grid_spacing: (float)
        The spacing between neighboring gridpoints.

    values: (1D np.array)
        The values of the probability density at all grid points.

    bounding_box:
        The domain in which the density is nonzero.

    Z:
        The normalization constant used to convert the field to a density.

    """

    def __init__(self, field_values, grid, bounding_box,
                 interpolation_method='cubic'):

        # Make sure grid and field are the same size
        self.field_values = field_values
        self.grid = grid
        self.bounding_box = bounding_box

        # Compute grid spacing
        self.grid_spacing = grid[1]-grid[0]

        # Compute normalization constant
        self.Z = np.sum(self.grid_spacing * np.exp(-self.field_values))

        # Interpolate using extended grid and extended phi
        self.field_func = interpolate.interp1d(self.grid,
                                               self.field_values,
                                               kind=interpolation_method,
                                               bounds_error=False,
                                               fill_value='extrapolate',
                                               assume_sorted=True)

        # Compute density values at supplied grid points
        self.values = self.evaluate(xs=self.grid)


    def evaluate(self, xs):
        """
        Evaluates the probability density at specified positions.

        Note: self(xs) can be used instead of self.evaluate(xs).

        Parameters
        ----------

        xs: (np.array)
            Locations at which to evaluate the density.

        Returns
        -------

        (np.array)
            Values of the density at the specified positions. Values at
            positions outside the bounding box are evaluated to zero.

        """

        values = np.exp(-self.field_func(xs)) / self.Z
        zero_indices = (xs < self.bounding_box[0]) | \
                       (xs > self.bounding_box[1])
        values[zero_indices] = 0.0
        return values

    def __call__(self, *args, **kwargs):
        """
        Same as evaluate()
        """

        return self.evaluate(*args, **kwargs)
