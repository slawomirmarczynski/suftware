"""
Functional Tests for Suftware
"""

# Standard modules
import numpy as np
import sys

# Import suftware 
sys.path.append('../')
import suftware as sw

# Generate datas
np.random.seed(0)
data = np.random.randn(100)

# simple test
sw.Density(data).plot() 
global_mistake = False
global_test_counter = 0


# helper method for functional test
def run_one_functional_test(**kw):
    
    functional_test = sw.Density(data,**kw)
    global global_mistake
    global_mistake = functional_test.mistake
    if global_mistake is True:
        display_status()
        sys.exit(1)
    global global_test_counter   
    global_test_counter += 1


# helper method for displaying pass/fail status
def display_status():
    if(global_mistake):
        print("Test Failed")
    else:
        print("Test Passed")


# grid Tests:

# values to be tested that are expected to fail
grid_fail_list = [5,'x',set(np.linspace(-3,3,100)),np.linspace(-3,3,5),
                      np.linspace(-3,3,1001),np.linspace(-1E-6,1E-6,100),
                     np.linspace(-1E6,1E6,100)]

# values to be tested that are expected to pass
grid_success_list = [None,np.linspace(-3,3,100),np.linspace(-3,3,100).T,np.matrix(np.linspace(-3,3,100)),
                    np.matrix(np.linspace(-3,3,100).T),list(np.linspace(-3,3,100)),np.linspace(-3,3,6),
                    np.linspace(-3,3,100),np.linspace(-3,3,100),np.linspace(-3,3,1000)]  
    
# should fail    
[run_one_functional_test(grid=grid_fail_list[i],should_fail=True) for i in range(len(grid_fail_list))]
# should pass
[run_one_functional_test(grid=grid_success_list[i],should_fail=False) for i in range(len(grid_success_list))]
    

display_status()

# grid_spacing Tests


# values to be tested
grid_spacing_fail_list = [0,0.0,-0.1,'0.1',[0.1],0.0001,1000.0]
grid_spacing_success_list = [None,0.02,0.1,0.5]

# should fail    
[run_one_functional_test(grid_spacing=grid_spacing_fail_list[i],should_fail=True) for i in range(len(grid_spacing_fail_list))]
# should pass
[run_one_functional_test(grid_spacing=grid_spacing_success_list[i],should_fail=False) for i in range(len(grid_spacing_success_list))]
    
display_status()    

# bounding_box

# values to be tested
bbox_fail_list = [{-6, 6},6,[6],[-6, 0, 6],['-6', '6'],[6, 6],[-1E-6, 1E-6],[-1E6, 1E6],[10, 20]]
bbox_success_list = [[-6, 6],(-6, 6),np.array([-6, 6]),[-.1, .1],[-10, 10]]

# should fail    
[run_one_functional_test(bounding_box=bbox_fail_list[i],should_fail=True) for i in range(len(bbox_fail_list))]
# should pass
[run_one_functional_test(bounding_box=bbox_success_list[i],should_fail=False) for i in range(len(bbox_success_list))]

display_status()


# num_grid_points

num_grid_points_fail_list = [-10,-1,0,1,2,3,4,5,1001] 
num_grid_points_success_list = [6,100,1000]

# should fail    
[run_one_functional_test(num_grid_points=num_grid_points_fail_list[i],should_fail=True) for i in range(len(num_grid_points_fail_list))]
# should pass
[run_one_functional_test(num_grid_points=num_grid_points_success_list[i],should_fail=False) for i in range(len(num_grid_points_success_list))]

display_status()

# alpha

# values to be tested
alpha_fail_list = [None,'x',-1,0.0,0,0.1,10]
alpha_success_list = [1,2,3,4]

# should fail    
[run_one_functional_test(alpha=alpha_fail_list[i],should_fail=True) for i in range(len(alpha_fail_list))]
# should pass
[run_one_functional_test(alpha=alpha_success_list[i],should_fail=False) for i in range(len(alpha_success_list))]

display_status()


# periodic 

# values to be tested
periodic_fail_list = [0,-1,'True','x',1]
perdiodic_success_list = [False,True]

# should fail    
[run_one_functional_test(periodic=periodic_fail_list[i],should_fail=True) for i in range(len(periodic_fail_list))]
# should pass
[run_one_functional_test(periodic=perdiodic_success_list[i],should_fail=False) for i in range(len(perdiodic_success_list))]

display_status()

# Z_evaluation_method

# values to be tested
Z_eval_fail_list = [0,'x','Einstein',False]
Z_eval_success_list = ['Lap','Lap+Fey','Lap+Imp']

# should fail    
#[run_one_functional_test(Z_evaluation_method=Z_eval_fail_list[i],should_fail=True) for i in range(len(Z_eval_fail_list))]
# should pass
#[run_one_functional_test(Z_evaluation_method=Z_eval_success_list[i],should_fail=False) for i in range(len(Z_eval_success_list))]

display_status()


# num_samples_for_Z

# Question: should a value of None pass or fail?

# values to be tested
num_samples_for_Z_fail_list = [None,-1,'x',0.1,1e5]
num_samples_for_Z_success_list =  [1,10,1000] 

# should fail    
[run_one_functional_test(num_samples_for_Z=num_samples_for_Z_fail_list[i],should_fail=True) for i in range(len(num_samples_for_Z_fail_list))]
# should pass
[run_one_functional_test(num_samples_for_Z=num_samples_for_Z_success_list[i],should_fail=False) for i in range(len(num_samples_for_Z_success_list))]

display_status()


global_test_counter
