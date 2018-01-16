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
sys.path.append('../code/')
import utils
TINY_FLOAT64 = sp.finfo(sp.float64).tiny

execfile('test_header.py')

num_trials = 1
N = 100

methods = [['Lap', 0, 'Lap', 10**4],
           ['Lap+Sam', 10**4, 'Lap', 10**4],
           ['GLap', 0, 'GLap', 10**3],
           ['GLap+Sam', 10**3, 'GLap', 10**3]]

data_seeds = np.random.randint(0, 10**8, num_trials)

num_methods = len(methods)
gds = np.zeros([num_trials, num_methods])
H_bias = np.zeros([num_trials, num_methods])
H_std = np.zeros([num_trials, num_methods])

for i in range(num_trials):

    print ''
    print '#', i+1
    data_seed = data_seeds[i]
    print 'data_seed =', data_seed

    for j in range(num_methods):

        Z_eval = methods[j][0]
        num_Z_samples = methods[j][1]
        pt_method = methods[j][2]
        num_pt_samples = methods[j][3]
        print ''
        print 'Running', Z_eval, '...'
        results = TestCase(N=N,
                           data_seed=data_seed,
                           G=100,
                           alpha=3,
                           bbox=[-6,6],
                           Z_eval=Z_eval,
                           num_Z_samples=num_Z_samples,
                           DT_MAX=0.5,  # <--- try this !
                           pt_method=pt_method,
                           num_pt_samples=num_pt_samples,
                           fix_t_at_t_star=True
                           ).run()

        xs = results.results.bin_centers
        Q_true = results.Q_true
        Q_star = results.results.Q_star
        Q_samples = results.results.Q_samples
        sample_weights = results.results.phi_weights
        gamma_prod = results.results.gamma_prod

        plt.plot(xs, Q_star, label=Z_eval)

        # Compute geo-distance between Q_star and Q_true
        gd = utils.geo_dist(Q_star, Q_true)
        gds[i,j] = gd

        # Compute entropy of Q_true and Q_samples
        h = xs[1] - xs[0]
        H_true = -sp.sum(Q_true * sp.log2(Q_true+TINY_FLOAT64)) * h
        H_samples = np.zeros(num_pt_samples)
        for k in range(num_pt_samples):
            Q_k = Q_samples[:,k]
            H_samples[k] = -sp.sum(Q_k * sp.log2(Q_k+TINY_FLOAT64)) * h
        H_sample_mean = sp.sum(H_samples * sample_weights) / sp.sum(sample_weights)
        H_sample_std = sp.sqrt(sp.sum(H_samples**2 * sample_weights) / sp.sum(sample_weights) - H_sample_mean**2)
        H_bias[i,j] = H_sample_mean - H_true
        H_std[i,j] = H_sample_std

    R = results.results.R
    plt.bar(xs, R, width=h, color='grey', alpha=0.5)
    plt.plot(xs, Q_true, color='black')
    plt.legend()
    plt.show()

# Plot 0: geo-distance vs method
plt.figure(0)
x = np.array([range(1, num_methods+1)]*num_trials)
for j in range(num_methods):
    plt.scatter(x[:,j], gds[:,j], color='black')
plt.xlabel('method')
plt.ylabel('geo-distance (rad)')
plt.savefig('gd')

success_rate = np.zeros(num_methods)
for j in range(num_methods):
    counter = 0.
    for i in range(num_trials):
        bias = H_bias[i,j]
        std = H_std[i,j]
        if abs(bias) < std:
            counter += 1
    success_rate[j] = counter / num_trials * 100

# Plot 1: H bias vs method
plt.figure(1)
x = np.array([range(1, num_methods+1)]*num_trials)
title_str = ''
for j in range(num_methods):
    plt.scatter(x[:,j], H_bias[:,j], color='black')
    title_str += '%s: %.1f%%  ' % (methods[j][0],success_rate[j])
plt.xlabel('method')
plt.ylabel('H bias (bits)')
plt.title(title_str)
plt.savefig('H_bias')

# Plot 2: H std vs method
plt.figure(2)
x = np.array([range(1, num_methods+1)]*num_trials)
for j in range(num_methods):
    plt.scatter(x[:,j], H_std[:,j], color='black')
plt.xlabel('method')
plt.ylabel('H std (bits)')
plt.title('H true = %.2f bits' % H_true)
plt.savefig('H_std')

print ''
print '--- done ---'
print
