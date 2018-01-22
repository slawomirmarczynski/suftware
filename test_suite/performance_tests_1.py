#!/usr/bin/env python

import scipy as sp
import numpy as np
from sklearn import mixture
from scipy.stats import gaussian_kde
import time
import matplotlib.pyplot as plt

from kinney2013_utils import make_gaussian_mix, draw_from_gaussian_mix, geo_dist

"""
NOTE: data and Q_true_func will be generated within deft. bbox can be specified by user.
      No interpolation will be performed.
"""

start_time = time.clock()

execfile('test_header.py')

np.random.seed(None)

# Specify parameters.
G = 100
alpha = 3
bbox = [-20,20]
Z_evals = ['Lap'] #, 'Lap+Sam', 'Lap+Fey', 'GLap+P', 'GLap+Sam+P']
DT_MAXs = [1.0] #, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]

# Specify Ns
Ns = [10000] #, 1000, 10000]
num_Ns = len(Ns)

# Number of trials to run
num_trials = 1

for i, N in enumerate(Ns):

    print('Performing trials for N = %d' % N)
    print('')

    for trial_num in range(num_trials):

        print('#%d' % trial_num)

        Q_star_all = np.zeros([len(Z_evals),G])
        t_star_all = np.zeros(len(Z_evals))
        gd_all = np.zeros(len(Z_evals))
        ts_all = np.zeros([len(Z_evals),200])
        log_Es_all = np.zeros([len(Z_evals),200])

        # Perform DEFT density estimation
        counter = 0
        for Z_eval in Z_evals:
            print('Z_eval = %s' % Z_eval)
            print('---')
            for DT_MAX in DT_MAXs:
                print('DT_MAX = %s' % DT_MAX)
                results = TestCase(N=N,
                                   G=G,
                                   alpha=alpha,
                                   bbox=bbox,
                                   Z_eval=Z_eval,
                                   DT_MAX=DT_MAX
                                   ).run()
                print('')
                ERROR_switch = results.ERROR_switch
                if not ERROR_switch:
                    break
            Q_star_all[counter,:] = results.results.Q_star
            t_star_all[counter] = results.results.t_star
            ts = [p.t for p in results.results.points]
            log_Es = [p.log_E for p in results.results.points]
            ts_all[counter,:len(ts)] = ts
            log_Es_all[counter,:len(ts)] = log_Es
            Q_true = results.Q_true
            Q_star = results.results.Q_star
            h = results.results.h
            gd_all[counter] = geo_dist(Q_true,Q_star,h)
            counter += 1

        # Make a plot of log_E and Q_star
        colors = ['blue', 'green', 'indigo', 'orange', 'red']
        linewidths = [1, 1, 1, 1, 1]
        plt.figure(figsize=[9,6])
        # plot Log_E
        plt.subplot(121)
        for k in range(len(Z_evals)):
            X = ts_all[k,:]
            x = X[X!=0]
            x = x[1:-1]
            Y = log_Es_all[k,:]
            y = Y[Y!=0]
            y = y[:-1]
            plt.plot(x, y, color=colors[k], linewidth=linewidths[k],
                     label=('t* = %.2f (%s)' % (t_star_all[k],Z_evals[k])))
        plt.xlabel('t')
        plt.ylabel('log_E')
        plt.legend()
        # Plot Q_star
        xs = results.results.bin_centers
        R = results.results.R
        h = results.results.h
        Q_true = results.Q_true
        plt.subplot(122)
        plt.bar(xs, R, width=h, color='grey', alpha=0.5)
        plt.plot(xs, Q_true, color='black', linewidth=1, label='Q_true')
        for k in range(len(Z_evals)):
            plt.plot(xs, Q_star_all[k,:], color=colors[k], linewidth=linewidths[k],
                     label=('gd = %.2f (%s)' % (gd_all[k],Z_evals[k])))
        plt.xlabel('x')
        plt.ylabel('Q')
        plt.legend()
        plt.show()

print('Took %.2f seconds to execute' % (time.clock()-start_time))
print('')
