from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import maxent
import utils
import simulate_data_1d
import scipy as sp

class TestMaxent(TestCase):

    def test_coeffs_to_field(self):
        pass


    def test_compute_maxent_prob_1d(self):
        # N number of data points
        N,G = 100, 10
        # why does a bounding box of [0,1] produce really negative phi's?
        bbox = [-6,6]
        alpha = 2
        # need to generate data to check compute maxent_field
        data, settings = simulate_data_1d.run('uniform', N)
        R, bin_centers = utils.histogram_counts_1d(data, G, bbox=bbox,normalized=True)
        # basis
        basis = utils.legendre_basis_1d(G,alpha)
        Q_actual = maxent.compute_maxent_prob_1d(R, basis)
        Q_expected = sp.array([ 0.07485046, 0.07955651,  0.08455844,  0.08987485,  0.09552553,  0.10153148, 0.10791503, 0.11469994,  0.12191144,  0.12957634])
        self.assertEquals(Q_actual.all(),Q_expected.all())