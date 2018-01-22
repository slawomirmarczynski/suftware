# Header cell
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Specific things needed
import time
import math
import sys
import pdb

# Add parent directory to path
sys.path.append('../code/')
sys.path.append('../sim/')

# Import deft modules
import deft_1d
import simulate_data_1d
import utils
from utils import DeftError

# Counts the number of test cases
global_counter = 0

class TestCase:
    ''' Contains information needed for a test run of deft_1d. '''
    
    def __init__(self, message, should_succeed, G=100, alpha=3, bbox=[-6,6], N=100, data_type='wide', 
                 num_samples=0, deft_seed=0, data_seed=0):
        
        # Set internal variables
        self.message = message
        self.G = G
        self.alpha = alpha
        self.bbox = bbox
        self.N = N
        self.data_type = data_type
        self.should_succeed = should_succeed
        self.num_samples = num_samples
        self.data_seed = data_seed
        self.deft_seed = deft_seed
        self.outcome_good = False

        # Increment test case num
        global global_counter
        self.test_num = global_counter
        global_counter += 1
        
        
    def __repr__(self):
        return 'G=%d, alpha=%d, bbox=%s, N=%d, data_type=%s, should_succeed=%s'%\
            (self.G, self.alpha, self.bbox, self.N, self.data_type, self.should_succeed)
        
    def run(self):
        
        # Set other variables
        periodic = False
        Laplace = False
        num_steps_per_sample = self.G
        num_thermalization_steps = 10*self.G
        fix_t_at_t_star = False
        print_t = False
        tollerance = 1E-3
        resolution = 1E-1
        
        # Generate data
        self.data, self.defaults = simulate_data_1d.run(self.data_type, self.N, self.data_seed)
        
        # Run deft_1d

        print '#%d'%self.test_num
        print self.message
        print self
        
        try:
            self.results = deft_1d.run(self.data, G=self.G, alpha=self.alpha, bbox=self.bbox, \
                                       periodic=periodic, Laplace=Laplace, num_samples=self.num_samples, \
                                       num_steps_per_sample=num_steps_per_sample, \
                                       num_thermalization_steps=num_thermalization_steps, \
                                       fix_t_at_t_star=fix_t_at_t_star, print_t=print_t, \
                                       tollerance=tollerance, resolution=resolution, seed=self.deft_seed)
            print 'Succeeded!'
            self.outcome_good = self.should_succeed

        except DeftError:
            print 'Failed in controlled way.'
            self.outcome_good = not self.should_succeed
            
        except:
            print 'Problem! Test failed in an uncontrolled way!'
            self.outcome_good = False
            
        if self.outcome_good:
            print 'Outcome: good.'
        else:
            print 'Outcome: BAD!'
        print ''
 

