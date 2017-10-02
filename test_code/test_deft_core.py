from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import numpy as np
import deft_core
import os
from scipy.sparse import csr_matrix

class TestDeftCore(TestCase):

    # setup replica class of laplacian as delta which will passed into deft_core.run to test functionality
    class Delta:

        _alpha = 3
        _kernel_dim = 3
        _kernel_basis = np.array([
            [1., -1.41421356, 1.14400963],
            [1., -0.70710678, -0.64661414],
            [1., 0., -1.24348873],
            [1., 0.70710678, -0.64661414],
            [1., 1.41421356, 1.14400963]])
        _eigenvalues = np.array([0., 0., 0., 78125., 546875.])

        delta = np.mat([
            [1., - 3., 3., - 1., 0.],
            [-3., 10., - 12., 6., - 1.],
            [3., - 12., 18., - 12., 3.],
            [-1., 6., - 12., 10., - 3.],
            [0., - 1., 3., - 3., 1.]])

        def get_G(self):
            # small value of G chosen for easy testing
            return 5

        def get_kernel_dim(self):
            return self._kernel_dim

        def get_kernel_basis(self):
            return self._kernel_basis

        def get_sparse_matrix(self):
            return csr_matrix(self.delta)

    def test_run(self):

        counts = np.array([0, 0, 2, 5, 3])
        delta = TestDeftCore.Delta()
        results = deft_core.run(counts,delta)
        predicted_Q_star = np.array([8.73962713e-05,9.40171643e-03,1.71301689e-01,5.28864295e-01, 2.90344903e-01])
        actual_Q_star = results.Q_star
        self.assertEqual(predicted_Q_star.all(),actual_Q_star.all())

