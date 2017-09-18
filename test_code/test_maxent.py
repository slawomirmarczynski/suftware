from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import maxent
import utils
import simulate_data_1d
import scipy as sp
import unittest

class TestMaxent(TestCase):


    # After basic unittest implementaion is completed, perform tests with other types of simulated data, e.g. uniform

    # set up some variables for re-use in the test methods below
    def setUp(self):
        # N number of data points
        self.N = 100
        self.G = 10
        # why does a bounding box of [0,1] produce really negative phi's?
        bbox = [-6,6]
        self.alpha = 2
        # need to generate data to check compute maxent_field
        self.data, self.settings = simulate_data_1d.run('uniform', self.N)
        self.R, self.bin_centers = utils.histogram_counts_1d(self.data, self.G, bbox=bbox,normalized=True)
        # basis/kernel
        self.basis = utils.legendre_basis_1d(self.G,self.alpha)
        self.coeffs = maxent.compute_maxent_field(self.R, self.basis)[0]

    def tearDown(self):
        pass

    # Begin tests

    # returns a raveled G by 1 column
    def test_coeffs_to_field(self):
        actual_col = maxent.coeffs_to_field(self.coeffs,self.basis)
        expected_col = sp.array([ 0.47197615,  0.41100069,  0.35002523,  0.28904977,  0.22807432,  0.16709886, 0.1061234, 0.04514794, -0.01582752, -0.07680298])
        self.assertEqual(actual_col.all(),expected_col.all())

    # test the action w.r.t field coefficients in a basis
    def test_action_per_datum_from_coeffs(self):
        actual_s = maxent.action_per_datum_from_coeffs(self.coeffs,self.R,self.basis)
        expected_s = 0.97260589603295966
        self.assertEqual(actual_s,expected_s)

    # test the action gradient w.r.t field coefficients in a basis
    def test_gradient_per_datum_from_coeffs(self):
        actual_action_gradient = maxent.gradient_per_datum_from_coeffs(self.coeffs,self.R,self.basis)
        expected_action_gradient = sp.array([-2.70935527e-07, 2.51591170e-07])
        self.assertEqual(actual_action_gradient.all(),expected_action_gradient.all())

    # test the action hessian w.r.t field coefficients in a basis
    def test_hessian_per_datum_from_coeffs(self):
        actual_action_hessian = maxent.hessian_per_datum_from_coeffs(self.coeffs,self.R,self.basis)
        expected_action_hessian = sp.array([[ 0.83345336,0.14504695],[ 0.14504695,0.84328002]])
        self.assertEqual(actual_action_hessian.all(),expected_action_hessian.all())

    # tests the maximum entropy probaiblity distribution in 1d
    def test_compute_maxent_prob_1d(self):

        Q_actual = maxent.compute_maxent_prob_1d(self.R, self.basis)
        Q_expected = sp.array([ 0.07485046, 0.07955651,  0.08455844,  0.08987485,  0.09552553,  0.10153148, 0.10791503, 0.11469994,  0.12191144,  0.12957634])
        self.assertEqual(Q_actual.all(),Q_expected.all())


    # test computed maxent field for all return types
    def test_compute_maxent_field(self):
        actual_coeffs, actual_phi, actual_success = maxent.compute_maxent_field(self.R, self.basis)
        expected_coeffs = sp.array([0.19758659, -0.17513867])
        expected_phi = sp.array([0.47197519,  0.41099986,  0.35002452,  0.28904919,  0.22807386,  0.16709853, 0.1061232, 0.04514787, -0.01582746, -0.07680279])
        expected_success = True
        self.assertEqual(actual_coeffs.all(),expected_coeffs.all())
        self.assertEqual(actual_phi.all(), expected_phi.all())
        self.assertEqual(actual_success,expected_success)


suite = unittest.TestLoader().loadTestsFromTestCase(TestMaxent)
unittest.TextTestRunner(verbosity=2).run(suite)