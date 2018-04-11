#!/usr/bin/env python
"""
Functional Tests for Suftware
"""

# Standard modules
import numpy as np
import sys

# Import suftware 
sys.path.append('../')
import suftware as sw

# Generate data
np.random.seed(0)
data = np.random.randn(100)

# simple test
density = sw.DensityEstimator(data)
global_mistake = False
global_test_success_counter = 0
global_test_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]

# helper method for functional test
def test(func, *args, **kw):
    functional_test = func(*args, **kw)
    global global_mistake
    global_mistake = functional_test.mistake
    if global_mistake is True:
        global global_test_fail_counter
        global_test_fail_counter += 1
    else:
        global global_test_success_counter
        global_test_success_counter += 1

# helper method for displaying pass/fail status
def display_local_status():
    print("Tests: passed: ", global_test_success_counter, ", tests failed: ", global_test_fail_counter,"\n")


def display_global_status():
    print('\033[1m' + "Total tests: passed: ", global_test_success_counter, " Total tests failed: ",
          global_test_fail_counter)

#
# sw.DensityEstimator.evaluate_samples()
#

def run_DensityEstimator_evaluate_samples_tests(var_name, fail_list,
                                                success_list):
    print("Testing '%s' parameter of " % var_name +
          "sw.DensityEstimator.evaluate_samples() ...")
    for x in fail_list:
        kw = {var_name: x}
        if 'x' not in kw:
            kw['x'] = 0
        test(func=density.evaluate_samples, should_fail=True, **kw)

    for x in success_list:
        kw = {var_name: x}
        if 'x' not in kw:
            kw['x'] = 0
        test(func=density.evaluate_samples, should_fail=False, **kw)

    display_local_status()


def test_DensityEstimator_evaluate_samples():
    run_DensityEstimator_evaluate_samples_tests(
        var_name='x',
        fail_list=[
            None,
            '1.0',
            1+2j,
            np.nan,
            np.Inf,
            {1:1, 2:2}.keys(),
            {1:1, 2:2}.values()
        ],
        success_list=[
            0,
            -1,
            1,
            1E6,
            range(10),
            np.random.randn(10),
            np.random.randn(3, 3),
            np.matrix(range(10)),
            np.matrix(range(10)).T,
            np.random.randn(2, 2, 2, 2)
        ]
    )

    run_DensityEstimator_evaluate_samples_tests(
        var_name='resample',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

#
# sw.DensityEstimator.evaluate()
#

def run_DensityEstimator_evaluate_tests(var_name, fail_list, success_list):
    print("Testing '%s' parameter of sw.DensityEstimator.evaluate() ..." \
          % var_name)
    for x in fail_list:
        kw = {var_name: x}
        test(func=density.evaluate, should_fail=True, **kw)
    for x in success_list:
        kw = {var_name: x}
        test(func=density.evaluate, should_fail=False, **kw)
    display_local_status()


def test_DensityEstimator_evaluate():
    run_DensityEstimator_evaluate_tests(
        var_name='x',
        fail_list=[
            None,
            '1.0',
            1+2j,
            np.nan,
            np.Inf,
            {1:1, 2:2}.keys(),
            {1:1, 2:2}.values()
        ],
        success_list=[
            0,
            -1,
            1,
            1E6,
            range(10),
            np.random.randn(10),
            np.random.randn(3, 3),
            np.matrix(range(10)),
            np.matrix(range(10)).T,
            np.random.randn(2, 2, 2, 2)
        ]
    )

#
# sw.DensityEstimator.get_stats()
#

def run_DensityEstimator_get_stats_tests(var_name, fail_list, success_list):
    print("Testing '%s' parameter of sw.DensityEstimator.get_stats() ..." \
          % var_name)
    for x in fail_list:
        kw = {var_name: x}
        test(func=density.get_stats, should_fail=True, **kw)
    for x in success_list:
        kw = {var_name: x}
        test(func=density.get_stats, should_fail=False, **kw)
    display_local_status()


def test_DensityEstimator_get_stats():
    run_DensityEstimator_get_stats_tests(
        var_name='use_weights',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

    run_DensityEstimator_get_stats_tests(
        var_name='show_samples',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )

#
# sw.DensityEstimator()
#

def run_DensityEstimator_tests(var_name, fail_list, success_list):
    print("Testing '%s' parameter of sw.DensityEstimator..." % var_name)
    for x in fail_list:
        kw = {var_name:x}
        if 'data' not in kw:
            kw['data'] = data
        test(func=sw.DensityEstimator, should_fail=True, **kw)
    for x in success_list:
        kw = {var_name:x}
        if 'data' not in kw:
            kw['data'] = data
        test(func=sw.DensityEstimator, should_fail=False, **kw)
    display_local_status()


def test_DensityEstimator_data():
    run_DensityEstimator_tests(
        var_name='data',
        fail_list=[
            None,
            5,
            [str(x) for x in data],
            [1]*5 + [2]*10,
            data.astype(complex)
        ],
        success_list=[
            data,
            np.random.randn(10),
            np.random.randn(int(1E6)),
            range(100),
            list(data),
            list(data) + [np.nan, np.Inf, -np.Inf],
            set(data)
        ]
    )


def test_DensityEstimator_grid():
    run_DensityEstimator_tests(
        var_name='grid',
        fail_list=[
            5,
            'x',
            set(np.linspace(-3, 3, 100)),
            np.linspace(-3, 3, 5),
            np.linspace(-3, 3, 1001),
            np.linspace(-1E-6, 1E-6, 100),
            np.linspace(-1E6, 1E6, 100)
        ],
        success_list=[
            None,
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 100).T,
            np.matrix(np.linspace(-3, 3, 100)),
            np.matrix(np.linspace(-3, 3, 100).T),
            list(np.linspace(-3, 3, 100)),
            np.linspace(-3, 3, 6),
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 100),
            np.linspace(-3, 3, 1000)
        ]
    )


def test_DensityEstimator_grid_spacing ():
    run_DensityEstimator_tests(
        var_name='grid_spacing',
        fail_list=[
            0,
            0.0,
            -0.1,
            '0.1',
            [0.1],
            0.0001,
            1000.0
        ],
        success_list=[
            None,
            0.05,
            0.1,
            0.5
        ]
    )


def test_DensityEstimator_bounding_box():
    run_DensityEstimator_tests(
        var_name='bounding_box',
        fail_list=[
            {-6, 6},
            6,
            [6],
            [-6, 0, 6],
            ['-6', '6'],
            [6, 6],
            [-1E-6, 1E-6],
            [-1E6, 1E6],
            [10, 20]
        ],
        success_list=[
            [-6, 6],
            (-6, 6),
            np.array([-6, 6]),
            [-.1, .1],
            [-10, 10]
        ]
    )


def test_DensityEstimator_num_grid_points():
    run_DensityEstimator_tests(
        var_name='num_grid_points',
        fail_list=[-10, -1, 0, 1, 2, 3, 4, 5, 1001],
        success_list=[6, 100, 1000]
    )


def test_DensityEstimator_alpha():
    run_DensityEstimator_tests(
        var_name='alpha',
        fail_list=[None, 'x', -1, 0.0, 0, 0.1, 10],
        success_list=[1, 2, 3, 4]
    )


def test_DensityEstimator_periodic():
    run_DensityEstimator_tests(
        var_name='periodic',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )


def test_DensityEstimator_evaluation_method_for_Z():
    run_DensityEstimator_tests(
        var_name='evaluation_method_for_Z',
        fail_list=[0, 'x', 'Einstein', False],
        success_list=['Lap', 'Lap+Fey', 'Lap+Imp']
    )


def test_DensityEstimator_num_samples_for_Z():
    run_DensityEstimator_tests(
        var_name='num_samples_for_Z',
        fail_list=[None, -1, 'x', 0.1, 1001],
        success_list=[0, 1, 10, 1000]
    )


def test_DensityEstimator_tolerance():
    run_DensityEstimator_tests(
        var_name='tolerance',
        fail_list=['x', -1, 0, 0.0],
        success_list=[1e-6, 1e-4, 1e-2, 1e-1, 1]
    )


def test_DensityEstimator_resolution():
    run_DensityEstimator_tests(
        var_name='resolution',
        fail_list=['x', -1, 0, 0.0, None],
        success_list = [1e-4, 1e-2, 1e-1, 1]
    )


def test_DensityEstimator_seed():
    run_DensityEstimator_tests(
        var_name='seed',
        fail_list=['x', 1e-5, 1.0, -1],
        success_list=[None, 1, 10, 100, 1000]
    )


def test_DensityEstimator_print_t():
    run_DensityEstimator_tests(
        var_name='print_t',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )


def test_DensityEstimator_num_posterior_samples():
    run_DensityEstimator_tests(
        var_name='num_posterior_samples',
        fail_list=['x', -1, 0.0, 1001],
        success_list=[0, 1, 2, 3, 10, 100, 1000]
    )


def test_DensityEstimator_sample_only_at_l_star():
    run_DensityEstimator_tests(
        var_name='sample_only_at_l_star',
        fail_list=bool_fail_list,
        success_list=bool_success_list
    )


def test_DensityEstimator_max_log_evidence_ratio_drop():
    run_DensityEstimator_tests(
        var_name='max_log_evidence_ratio_drop',
        fail_list=['x', -1, 0, None, 0],
        success_list=[0.1, 1, 2, 3, 10, 100, 100.0, 1000]
    )


# Run functional tests
if __name__ == '__main__':

    # DensityEstimator constructor
    test_DensityEstimator_data()
    test_DensityEstimator_grid()
    test_DensityEstimator_grid_spacing()
    test_DensityEstimator_bounding_box()
    test_DensityEstimator_num_grid_points()
    test_DensityEstimator_alpha()
    test_DensityEstimator_periodic()
    test_DensityEstimator_evaluation_method_for_Z()
    test_DensityEstimator_num_samples_for_Z()
    test_DensityEstimator_tolerance()
    test_DensityEstimator_resolution()
    test_DensityEstimator_seed()
    test_DensityEstimator_print_t()
    test_DensityEstimator_num_posterior_samples()
    test_DensityEstimator_sample_only_at_l_star()
    test_DensityEstimator_max_log_evidence_ratio_drop()

    # DensityEstimator methods
    test_DensityEstimator_get_stats()
    test_DensityEstimator_evaluate()
    test_DensityEstimator_evaluate_samples()

    # Print results
    display_global_status()
