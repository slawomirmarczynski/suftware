#!/usr/bin/env python

import scipy as sp
import numpy as np
from sklearn import mixture
from scipy.stats import gaussian_kde
import time
import matplotlib.pyplot as plt

from kinney2013_utils import make_gaussian_mix, draw_from_gaussian_mix, geo_dist

"""
NOTE: data, bbox, and Q_true_func will be generated from a gaussian mixture and fed into deft.
      Densities will be refined by interpolation, so bbox cannot be specified by user.
"""

start_time = time.clock()

execfile('test_header.py')

np.random.seed(None)

# Specify parameters
G = 100
alpha = 3
Z_evals = ['Lap', 'Lap+Sam', 'Lap+Fey', 'GLap+P', 'GLap+Sam+P']
DT_MAXs = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]

# Specify Ns
Ns = [100] #, 1000, 10000]
num_Ns = len(Ns)

# Number of gaussians
num_gaussians = 5

# Number of gridpoints used to refine the densities
plot_grid_pts = 1000

# Number of trials to run
num_trials = 10

for i, N in enumerate(Ns):

    print('Performing trials for N = %d' % N)
    print('')

    for trial_num in range(num_trials):

        print('#%d' % trial_num)

        # Choose mixture of gaussians
        gaussians = make_gaussian_mix(num_gaussians)
        # Draw data from mixture
        [xis, xgrid, Q_true_func, details] = draw_from_gaussian_mix(N=N, Nx=G, gaussians=gaussians)
        # Compute data range and grid for fine-grained analysis
        xmin = min(xgrid)
        xmax = max(xgrid)
        xint = [xmin, xmax]
        xs = sp.linspace(xmin, xmax, plot_grid_pts)
        dx = xs[1] - xs[0]
        # Interpolate Q_true for plotting
        Q_true_vals = Q_true_func(xs)

        Q_star_vals_all = np.zeros([len(Z_evals),plot_grid_pts])
        t_star_all = np.zeros(len(Z_evals))
        ts_all = np.zeros([len(Z_evals),200])
        log_Es_all = np.zeros([len(Z_evals),200])
        gd_all = np.zeros(len(Z_evals))

        # Perform DEFT density estimation
        counter = 0
        for Z_eval in Z_evals:
            print('Z_eval = %s' % Z_eval)
            print('---')
            for DT_MAX in DT_MAXs:
                print('DT_MAX = %s' % DT_MAX)
                results = TestCase(feed_data=True,
                                   data_fed=xis,
                                   Q_true_func=Q_true_func,
                                   N=N,
                                   G=G,
                                   alpha=alpha,
                                   bbox=xint,
                                   Z_eval=Z_eval,
                                   DT_MAX=DT_MAX
                                   ).run()
                print('')
                ERROR_switch = results.ERROR_switch
                if not ERROR_switch:
                    break
            Q_star_vals = results.results.Q_star_func(xs)
            Q_star_vals_all[counter,:] = Q_star_vals[:]
            t_star_all[counter] = results.results.t_star
            ts = [p.t for p in results.results.points]
            log_Es = [p.log_E for p in results.results.points]
            ts_all[counter,:len(ts)] = ts
            log_Es_all[counter,:len(ts)] = log_Es
            gd_all[counter] = geo_dist(Q_true_vals, Q_star_vals, dx)
            counter += 1

        # Make a plot of log_E and Q_star
        colors = ['blue', 'green', 'purple', 'orange', 'red']
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
        bc = results.results.bin_centers
        R = results.results.R
        h = results.results.h
        plt.subplot(122)
        plt.bar(bc, R, width=h, color='grey', alpha=0.5)
        plt.plot(xs, Q_true_vals, color='black', linewidth=1, label='Q_true')
        for k in range(len(Z_evals)):
            plt.plot(xs, Q_star_vals_all[k,:], color=colors[k], linewidth=linewidths[k],
                     label=('gd = %.2f (%s)' % (gd_all[k],Z_evals[k])))
        plt.xlabel('x')
        plt.ylabel('Q')
        plt.legend()
        plt.show()

print('Took %.2f seconds to execute' % (time.clock()-start_time))
print('')
