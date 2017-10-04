from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import scipy as sp
import numpy as np
import deft_core
import unittest
from scipy.sparse import csr_matrix

class TestDeftCore(TestCase):

    # the following two methods run before everything and after anything, in the unit test suite. Objects with suite wide scope may be defined here for better performance.
    @classmethod
    def setUpClass(cls):

        cls.phi_infty = np.array([ 7.59930786,  3.01371323,  0.15648113, -0.97238842, -0.37289543])
        cls.R = np.array([ 0., 0., 0.2,  0.5,  0.3])
        cls.N = 10
        cls.delta = TestDeftCore.Delta()

    @classmethod
    def tearDownClass(cls):
        pass

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
        delta = self.delta
        results = deft_core.run(counts,delta)
        predicted_Q_star = np.array([8.7396271344805499e-05, 9.4017164259393576e-03, 1.7130168936492607e-01, 5.2886429495267018e-01, 2.9034490298511967e-01])
        actual_Q_star = results.Q_star
        predicted_R = np.array([0.,0.,0.2,0.5,0.3])
        actual_R = results.R
        self.assertEqual(predicted_Q_star.tolist(),actual_Q_star.tolist())
        self.assertEqual(predicted_R.tolist(), actual_R.tolist())

    def test_compute_map_curve(self):
        map_curve_actual = deft_core.compute_map_curve(self.N,self.R,self.delta)
        actual_t = sp.array([p.t for p in map_curve_actual.points])
        predicted_t = np.array([-np.inf,   0.,                  1.,                  2.,
                                2.5392220639675585,   2.9299162740567244,   3.2584596317615167,
                                3.5542893085059477,   3.8319910581692125,   4.1003895258805398,
                                4.3656622894509312,   4.6327052332959937,   4.9058578556237924,
                                5.1893734949820542,   5.4878039456936829,   5.8064002819804124,
                                6.1516242499409159,   6.5319014009163876,   6.9588494544043584,
                                7.449460625415953,    8.0303352276425244,   8.7468058319425737,
                                9.6855280829599746,  10.6855280829599746,   11.6855280829599746,
                                np.inf])
        self.assertEqual(actual_t.tolist(),predicted_t.tolist())

    def test_compute_corrector_step(self):
        phiNew_corrector_step_t0_actual = deft_core.compute_corrector_step(self.phi_infty,self.R,self.delta,0,self.N)
        phiNew_corrector_step_t0_predicted = np.array([7.735609670513095,3.0574147401594662,0.1548807439299333,-0.972424856445949, - 0.372762525711791])
        self.assertEqual(phiNew_corrector_step_t0_actual.tolist(),phiNew_corrector_step_t0_predicted.tolist())

    def test_compute_predictor_step(self):
        # choosing values used in deft core, should also use other arbitrary values
        resolution = 1E-2
        direction = -1
        predictor_step_actual, t_new = deft_core.compute_predictor_step(self.phi_infty,self.R,self.delta,0,self.N,direction,resolution)
        predictor_step_predicted = np.array([7.4631945039611409,  2.9702001688713144,  0.15826997173369853,-0.9721635291813676, -0.37283987940842334])
        self.assertEqual(predictor_step_actual.tolist(),predictor_step_predicted.tolist())


suite = unittest.TestLoader().loadTestsFromTestCase(TestDeftCore)
unittest.TextTestRunner(verbosity=2).run(suite)