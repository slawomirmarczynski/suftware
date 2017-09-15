from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import maxent
import utils
import simulate_data_1d
import scipy as sp

class TestMaxent(TestCase):


    # set up some variables for re-use in the test methods
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

    def tearDown(self):
        pass

    # Begin tests

    def test_coeffs_to_field(self):
        pass

    # tests the maximum entropy probaiblity distribution in 1d
    def test_compute_maxent_prob_1d(self):

        Q_actual = maxent.compute_maxent_prob_1d(self.R, self.basis)
        Q_expected = sp.array([ 0.07485046, 0.07955651,  0.08455844,  0.08987485,  0.09552553,  0.10153148, 0.10791503, 0.11469994,  0.12191144,  0.12957634])
        self.assertEquals(Q_actual.all(),Q_expected.all())
        #maxent.compute_maxent_field(R, basis)


    # test computed maxent field for all return types
    def test_compute_maxent_field(self):
        actual_coeffs, actual_phi, actual_success = maxent.compute_maxent_field(self.R, self.basis)
        expected_coeffs = sp.array([0.19758659, -0.17513867])
        expected_phi = sp.array([0.47197519,  0.41099986,  0.35002452,  0.28904919,  0.22807386,  0.16709853, 0.1061232, 0.04514787, -0.01582746, -0.07680279])
        expected_success = True
        self.assertEquals(actual_coeffs.all(),expected_coeffs.all())
        self.assertEquals(actual_phi.all(), expected_phi.all())
        self.assertEquals(actual_success,expected_success)

