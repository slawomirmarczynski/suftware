import sys
sys.path.append('../code')

from unittest import TestCase
# for the methods in the Laplacian class
from laplacian import Laplacian
# to test the rest of the methods in the laplacian file
import laplacian
# for testing the save method in the laplacian file
from mock import patch
# for testing test derivatives and matrices (scipy objects)
import scipy as sp
import unittest
import os

class TestLaplacian(TestCase):


    # the following two methods run before everything and after anything, in the unit test suite. Objects with suite wide scope may be defined here for better performance.
    @classmethod
    def setUpClass(cls):
        print('setup class ')

    @classmethod
    def tearDownClass(cls):
        print(' tear down class ')

    # the following two methods run before and every unit test. They will be helpful in keeping the tester method clear and less cluttered
    def setUp(self):
        pass

    def tearDown(self):
        pass

    """
    This method tests behaviour of the laplacian constructor when invalid arguments are passed. This method needs to pass to ensure that accurate
    error handling is present in the constructor to laplacian.py.
    A simple example usage: operator order should >= 1, (amongst other things like being an integer, not null, and being in integer format)
    if this test fails, that means that some parameter was accurately exception handled. While thisn't testing functionality of a class, it serves to build extra robustness
    and confidence in software.
    """
    def test_IllegalParametersInLaplacianConstructor(self):
        # illegal operator order parameter values
        operator_order_value_zero = 0
        operator_order_floating_format = 1.2
        """
        Args:   operator_type (str): The type of operator. Accepts one of the following values:
                    '1d_bilateral'
                    '1d_periodic'
                    '2d_bilateral'
                    '2d_periodic'

                operator_order (int):
                    The order of the operator.

                num_gridpoints:
                    The number of gridpoints in each dimension of the domain.
        """
        # I expect a value error for a operator_order parameter less than 1
        self.failUnlessRaises(ValueError,Laplacian,'1d_bilateral', operator_order_value_zero, 20, 1.0)
        # I expect a value error for a operator_order parameter of type floating point
        self.failUnlessRaises(ValueError, Laplacian, '1d_bilateral', operator_order_floating_format, 20, 1.0)

        # similar checks need to be implemented for null values and other parameters

    #Test that the returned a (G-1)xG sized 1d derivative matrix for a particular value of G equates with the expected value
    def test_derivative_matrix_1d(self):
        expectedMatrixForGEqual4 = sp.matrix('1. 1. 0. 0.; 0. -1.  1.  0.;0.  0. -1.  1.')
        actualMatrixForGEqual4 = laplacian.derivative_matrix_1d(4)
        self.assertEqual(expectedMatrixForGEqual4.all(),actualMatrixForGEqual4.all())

    # return of laplacian_1d is a GxG sized 1d bilateral laplacian matrix of order alpha; this test checks and expected value and an edge case
    def test_laplacian_1d(self):

        # a test with default parameters
        actualMatrixForGEqual3 = laplacian.laplacian_1d(3,1,1.0)
        expectedMatrixForGEqual3 = sp.matrix('1. -1.  0.;-1.  2. -1.;0. -1.  1.')
        self.assertEqual(expectedMatrixForGEqual3.all(), actualMatrixForGEqual3.all())

        # test with periodic == true
        actualMatrixForGEqual3PeriodicEqualTrue = laplacian.laplacian_1d(3, 1, 1.0,periodic=True)
        expectedMatrixForGEqual3PeriodicEqualTrue = sp.matrix('2. -1. -1.;-1.  2. -1.;-1. -1.  2.')
        self.assertEquals(expectedMatrixForGEqual3PeriodicEqualTrue.all(),actualMatrixForGEqual3PeriodicEqualTrue.all())

        # test all other scenarios

    # This test ensures that dump gets called in the save method in the Laplacian class. This test will fail if functionality of save method in Laplacian is modified.
    @patch('pickle.dump')
    def test_saveMethodInLaplacianFile(self,dump):
        # create laplacian object
        l = Laplacian('1d_bilateral', 1, 20, 1.0)
        # call save
        l.save('foo.pkl')
        # ensure pickle dump happens
        self.assertTrue(dump.called)
        # delete dummy file
        os.remove('foo.pkl')

        #pass

    # In general, unit tests should not be written for getter and setter methods (according to good programming standards), however they are left as is for 100% code coverage for
    # illustrative purposes. A test may be written for any of these methods if it proves to be helpful.
    def test_get_G(self):
        # a dummy test could be that the returned G could be asserted Equal to an expected G. But the state of G may be different during different periods in runtime, so
        # this will likely not be a useful test.
        pass

    def test_get_kernel_basis(self):
        pass

    def test_get_kernel_dim(self):
        pass

    def test_get_sparse_matrix(self):
        pass

    def test_get_sparse_Lambda(self):
        pass

    def test_get_dense_Lambda(self):
        pass

    def test_get_dense_matrix(self):
        pass

suite = unittest.TestLoader().loadTestsFromTestCase(TestLaplacian)
unittest.TextTestRunner(verbosity=2).run(suite)