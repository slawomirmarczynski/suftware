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
            'Lap[+P]'   : sampling from Laplace approximation
            'Lap+W[+P]' : sampling from Laplace approximation with importance weights
            'GLap[+P]'  : sampling from generalized Laplace approximation
            'GLap+W[+P]': sampling from generalized Laplace approximation with importance weights
            'MMC'       : sampling using Metropolis Monte Carlo

Note 1: [+P] means this task can be done in parallel
Note 2: importance weights partly depend on how the partition function is evaluated
"""
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

execfile('test_header.py')

num_trials = 1
num_powers = 5

data_size = np.zeros([num_trials,num_powers])
data_slope = np.zeros([num_trials,num_powers])
data_mean = np.zeros([num_trials,num_powers])
data_mean_std = np.zeros([num_trials,num_powers])
data_Z_ratio = np.zeros([num_trials,num_powers])
data_l_mean = np.zeros([num_trials,num_powers])
data_l_std = np.zeros([num_trials,num_powers])
data_r_mean = np.zeros([num_trials,num_powers])
data_r_std = np.zeros([num_trials,num_powers])

for i in range(num_trials):
    print ''
    print '#', i
    num_pt_samples = 10**num_powers
    results = TestCase(N=100,
                       G=100,
                       data_seed=None,
                       DT_MAX=1.0,
                       alpha=3,
                       Z_eval='Lap',
                       pt_method='Lap',
                       fix_t_at_t_star=True,
                       num_pt_samples=num_pt_samples,
                       bbox=[-6,6]  # change indices of l & r peaks below accordingly
                       ).run()

    phi_weights = results.results.phi_weights
    gamma_prod = results.results.gamma_prod
    Q_samples = results.results.Q_samples
    Q_star = results.results.Q_star
    Q_true = results.Q_true
    xs = results.results.bin_centers

    for p in range(num_powers):

        num_samples = 10**(p+1)
        weights = phi_weights[0:num_samples]
        l_peaks = Q_samples[33,0:num_samples]  # <--- index of the left peak
        r_peaks = Q_samples[66,0:num_samples]  # <--- index of the right peak
        # a. bbox=[ -6, 6] -> 33 & 66
        # b. bbox=[-18,18] -> 44 & 55
        # c. bbox=[-18, 6] -> 66 & 83
        # d. bbox=[ -3, 3] -> 16 & 83

        # Compute effective sample size in unit of %
        eff_sample_size = sp.sum(weights)**2 / sp.sum(weights**2) / num_samples * 100
        data_size[i,p] = eff_sample_size

        # Compute slope of the log distribution of the weights near the tail
        if num_samples <= 10**3:
            num_bins = 50
        else:
            num_bins = num_pt_samples / 100
        hist, bin_edges = np.histogram(weights, bins=num_bins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = sp.linspace(bin_edges[0]+0.5*bin_width, bin_edges[-1]-0.5*bin_width, num_bins)
        indices = sp.array(np.nonzero(hist)).ravel()
        x0 = sp.log(bin_centers[indices])
        y0 = sp.log(hist[indices])
        x = x0[y0.argmax():]
        y = y0[y0.argmax():]
        z = np.polyfit(x, y, 1)
        slope = z[0]
        data_slope[i,p] = slope

        # Compute mean of the weights and its std
        sample_mean = sp.mean(weights)
        sample_mean_std = sp.std(weights) / sp.sqrt(num_samples)
        data_mean[i,p] = sample_mean
        data_mean_std[i,p] = sample_mean_std

        # Compute ratio of Z to Z_lap
        Z_ratio = gamma_prod * sample_mean
        data_Z_ratio[i,p] = Z_ratio

        # Compute weighted mean and std of Q samples at the left peak
        l_mean = sp.sum(l_peaks * weights) / sp.sum(weights)
        l_std = sp.sqrt(sp.sum(l_peaks**2 * weights) / sp.sum(weights) - l_mean**2)
        data_l_mean[i,p] = l_mean
        data_l_std[i,p] = l_std

        # Compute weighted mean and std of Q samples at the right peak
        r_mean = sp.sum(r_peaks * weights) / sp.sum(weights)
        r_std = sp.sqrt(sp.sum(r_peaks**2 * weights) / sp.sum(weights) - r_mean**2)
        data_r_mean[i,p] = r_mean
        data_r_std[i,p] = r_std

        print l_mean, l_std
        print r_mean, r_std
        print '---'
        """
        plt.plot(xs, Q_samples[:,0:num_samples], color='blue', alpha=0.5)
        plt.plot(xs, Q_star, color='red')
        plt.show()
        """

        # Plot distribution of Q samples at the left peak. Used for a particular problem
        peaks = l_peaks
        num_bins = 30
        p_min = peaks.min()
        p_max = peaks.max()
        spread = p_max - p_min
        lb = p_min  # - 0.2*spread
        ub = p_max  # + 0.2*spread
        bin_edges = np.linspace(lb, ub, num_bins+1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = np.linspace(lb+0.5*bin_width, ub-0.5*bin_width, num_bins)
        hist = np.zeros(num_bins)
        for k in range(num_samples):
            p_k = peaks[k]
            for j in range(num_bins):
                if (p_k-bin_edges[j]) * (p_k-bin_edges[j+1]) < 0:
                    hist[j] += weights[k]
                    break
        hist = hist / sp.sum(hist)
        #plt.bar(bin_centers, hist, width=bin_width, color='grey', alpha=0.5, edgecolor='grey')
        hist_cspline = interp1d(bin_centers, hist, kind='cubic')
        bin_centers = np.linspace(lb+0.5*bin_width, ub-0.5*bin_width, 5*num_bins)
        plt.plot(bin_centers, hist_cspline(bin_centers), label=(p+1))

plt.ylim(0, 0.2)
plt.legend()
plt.show()

print '---'
raise

# Plot 1: effective sample size (%) vs log N
plt.figure(1)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_size[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('effective sample size (%)')
plt.ylim(0, 100)
plt.savefig('eff_sample_size')

# Plot 2: slope vs log N
plt.figure(2)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_slope[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('slope')
plt.ylim(-6, 0)
plt.savefig('slope')

# Plot 3: sample mean vs log N
plt.figure(3)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_mean[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('sample mean')
plt.ylim(0, 3)
plt.savefig('sample_mean')

# Plot 4: sample mean std vs log N
plt.figure(4)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_mean_std[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('sample mean std')
plt.ylim(-0.5, 1.5)
plt.savefig('sample_mean_std')

# Plot 5: Z / Z_lap vs log N
plt.figure(5)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_Z_ratio[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('Z / Z_lap')
plt.ylim(0, 2)
plt.savefig('Z_ratio')

# Plot 6: l-peak mean vs log N
plt.figure(6)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_l_mean[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('l-peak mean')
#plt.ylim(0, 2)
plt.savefig('l_peak_mean')

# Plot 7: l-peak std vs log N
plt.figure(7)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_l_std[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('l-peak std')
#plt.ylim(0, 2)
plt.savefig('l_peak_std')

# Plot 8: r-peak mean vs log N
plt.figure(8)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_r_mean[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('r-peak mean')
#plt.ylim(0, 2)
plt.savefig('r_peak_mean')

# Plot 9: r-peak std vs log N
plt.figure(9)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, data_r_std[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('r-peak std')
#plt.ylim(0, 2)
plt.savefig('r_peak_std')

print ''
print '--- ok ---'
print ''
