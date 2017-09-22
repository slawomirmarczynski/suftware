from unittest import TestCase

import sys
sys.path.append('../code')
sys.path.append('../sim/')

import numpy as np
import unittest
import deft_1d
import simulate_data_1d

class TestDeft1d(TestCase):

    def setUp(self):
        self.N = 10
        #both of the following lines should work
        #self.data = simulate_data_1d.run('uniform', N=self.N)
        self.data, self.settings = simulate_data_1d.run("uniform",N=self.N)

    def tearDown(self):
        pass

    # method that checks the main calculation of deft_1d by calling run and ensuring that we get the correct Q_star
    def test_run(self):
        actual_Q_star = deft_1d.run(self.data).Q_star
        expected_Q_star = np.array([0.26208081,  0.27902308,  0.29671363,  0.31515726,  0.33435637,  0.35431076, 0.3750175,   0.39647079,  0.41866178,  0.44157847,  0.46520559,  0.48952448,0.51451301,  0.54014552,  0.5663927,   0.59322163,  0.6205957,   0.64847466,0.67681461, 0.70556803,  0.73468391,  0.76410778,  0.79378187,  0.82364525,0.85363396,  0.88368123,  0.91371771,  0.94367169,  0.97346933,  1.003035,1.03229154,  1.06116061,  1.08956298,  1.11741895,  1.14464863,  1.17117239,1.1969112,  1.22178702,  1.24572322,  1.26864491,  1.29047941,  1.31115657,1.33060914,  1.3487732,   1.36558844,  1.38099851,  1.39495137,  1.40739953,1.41830036, 1.42761632,  1.43531517, 1.44137014,  1.44576015,  1.44846983,1.44948972,  1.44881623,  1.44645172,  1.44240448,  1.43668865,  1.42932417,  1.42033665,  1.40975722,  1.39762235,  1.38397366,  1.36885765,  1.35232547,1.33443261,  1.31523857,  1.2948066,   1.27320326,  1.25049816,  1.2267635,1.20207372,  1.17650515,  1.15013557,  1.12304382,  1.09530947,  1.06701236,1.03823227,  1.00904855,  0.97953976,  0.94978332,  0.91985521,  0.88982965,0.8597788,   0.82977253,  0.79987816,  0.77016022,  0.7406803,   0.71149681,0.68266492,  0.65423635,  0.62625935,  0.59877854,  0.57183495,  0.54546593,0.51970514,  0.49458263,  0.47012481,  0.44635452,])
        self.assertEqual(actual_Q_star.all(),expected_Q_star.all())

    # helper method for test_get_data_file_hand()
    def raiseFileNotFoundError(self):
        return FileNotFoundError


    # this test ensures that loading data has appropriate exception handling
    def test_get_data_file_handle(self):
        # call function with a dummy arguement and check if the appropriate exception gets called
        dummyFile = 'dummyFileName'
        x = deft_1d.get_data_file_handle(dummyFile)
        self.assertEqual(type(x), self.raiseFileNotFoundError())

suite = unittest.TestLoader().loadTestsFromTestCase(TestDeft1d)
unittest.TextTestRunner(verbosity=2).run(suite)