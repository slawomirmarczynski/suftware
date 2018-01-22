
# coding: utf-8

# In[1]:


#!/usr/bin/env python
"""
Ways to evaluate partition function
Z_eval = 'Lap'         : Laplace approximation (default)
         'Lap+Sam[+P]' : Laplace approximation + importance sampling
         'GLap[+P]'    : generalized Laplace approximation
         'GLap+Sam[+P]': generalized Laplace approximation + importance sampling
         'Lap+Fey'     : Laplace approximation + Feynman diagrams

Methods of posterior sampling
pt_method =  None       : no sampling will be performed (default)
            'Lap[+P]'   : sampling from Laplace approximation + importance weight
            'GLap[+P]'  : sampling from generalized Laplace approximation + importance weight
            'MMC'       : sampling using Metropolis Monte Carlo

Note: [+P] means this task can be done in parallel
"""
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append('deft_code/')
TINY_FLOAT64 = sp.finfo(sp.float64).tiny
import os

#execfile('test_header.py')
exec(open(os.getcwd()+"/test_suite/test_header.py").read())

N = 100
bbox = [-15,15]
Z_eval = 'Lap'
num_Z_samples = 0
pt_method = 'Lap'
num_pt_samples = 100
fix_t_at_t_star = True
# make this pickalable
results = TestCase(N=N,data_seed=0,deft_seed=0,G=100,alpha=3,bbox=bbox,Z_eval=Z_eval,num_Z_samples=num_Z_samples,DT_MAX=1.0,pt_method=pt_method,num_pt_samples=num_pt_samples,fix_t_at_t_star=fix_t_at_t_star).run()

print(results.results.l_star)

xs = results.results.bin_centers
phi_samples = results.results.phi_samples
phi_star = results.results.phi_star

'''
plt.plot(xs, phi_samples, color='blue', alpha=0.3)
plt.plot(xs, phi_star, color='red')
plt.ylim(-5,5)
plt.show()
'''

sample_weights = results.results.phi_weights
indices = range(num_pt_samples)
index_probs = sample_weights / sum(sample_weights)
weighted_sample_indices = np.random.choice(indices, size=num_pt_samples, p=index_probs)
phi_samples_weighted = phi_samples[:,weighted_sample_indices]

'''
plt.plot(xs, phi_samples_weighted, color='blue', alpha=0.3)
plt.plot(xs, phi_star, color='red')
plt.ylim(-5,5)
plt.show()
'''


# Naive Laplace sampling 
xs = results.results.bin_centers
R = results.results.R
h = results.results.h
Q_true = results.Q_true
Q_star = results.results.Q_star
Q_samples = results.results.Q_samples
sample_weights = results.results.phi_weights

plt.figure(1)
plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
plt.plot(xs, Q_true, color='black', zorder=3)
plt.plot(xs, Q_star, color='red', zorder=4)
plt.plot(xs, Q_samples, color='blue', alpha=0.3, zorder=1)
plt.ylim(0, 0.4)
plt.show()


# In[10]:


# Re-sampling according to importance weights
indices = range(num_pt_samples)
index_probs = sample_weights / sum(sample_weights)
weighted_sample_indices = np.random.choice(indices, size=num_pt_samples, p=index_probs)
Q_samples_weighted = Q_samples[:,weighted_sample_indices]

'''
plt.figure(2)
plt.bar(xs, R, width=h, color='grey', alpha=0.3, zorder=2)
plt.plot(xs, Q_true, color='black', zorder=3)
plt.plot(xs, Q_star, color='red', zorder=4)
plt.plot(xs, Q_samples_weighted, color='blue', alpha=0.2, zorder=1)
plt.ylim(0, 0.4)
plt.show()
'''

