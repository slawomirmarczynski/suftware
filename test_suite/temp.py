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
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
TINY_FLOAT64 = sp.finfo(sp.float64).tiny

execfile('test_header.py')

num_trials = 50
num_powers = 6

num_pt_samples = 10**num_powers

w_mean = np.zeros([num_trials,num_powers])
w_mean_std = np.zeros([num_trials,num_powers])
H_mean = np.zeros([num_trials,num_powers])
H_mean_std = np.zeros([num_trials,num_powers])

for i in range(num_trials):
    print ''
    print '#', i+1
    results = TestCase(N=50,
                       data_seed=None,
                       G=100,
                       alpha=3,
                       bbox=[-6,6],
                       Z_eval='GLap',
                       num_Z_samples=0,
                       DT_MAX=1.0,
                       pt_method='GLap',
                       num_pt_samples=num_pt_samples,
                       fix_t_at_t_star=True
                       ).run()

    xs = results.results.bin_centers
    Q_true = results.Q_true
    Q_star = results.results.Q_star
    Q_samples = results.results.Q_samples
    sample_weights = results.results.phi_weights
    gamma_prod = results.results.gamma_prod

    # Compute entropy of Q_true and Q_samples
    h = xs[1] - xs[0]
    H_true = -sp.sum(Q_true * sp.log2(Q_true + TINY_FLOAT64)) * h
    H_samples = np.zeros(num_pt_samples)
    for k in range(num_pt_samples):
        Q_k = Q_samples[:,k]
        H_samples[k] = -sp.sum(Q_k * sp.log2(Q_k + TINY_FLOAT64)) * h

    for p in range(num_powers):
        num_samples = 10**(p+1)
        weights = sample_weights[0:num_samples]
        entropies = H_samples[0:num_samples]
        # Compute sample mean of the weights and its std
        w_sample_mean = sp.mean(weights)
        w_sample_mean_std = sp.std(weights) / sp.sqrt(num_samples)
        w_mean[i,p] = w_sample_mean
        w_mean_std[i,p] = w_sample_mean_std
        # Compute sample mean of the entropies and its std
        H_sample_mean = sp.sum(entropies * weights) / sp.sum(weights)
        H_sample_std = sp.sqrt(sp.sum(entropies**2 * weights) / sp.sum(weights) - H_sample_mean**2)
        H_sample_mean_std = H_sample_std / sp.sqrt(num_samples)
        H_mean[i,p] = H_sample_mean
        H_mean_std[i,p] = H_sample_mean_std

# Plot 1: w mean vs log N
plt.figure(1)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, w_mean[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('w mean')
plt.ylim(0.5,1.5)
plt.grid(axis='y', linestyle='--', color='grey')
plt.savefig('w_mean')

# Plot 2: w mean std vs log N
plt.figure(2)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, w_mean_std[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('w mean std')
plt.ylim(0.0,0.5)
plt.grid(axis='y', linestyle='--', color='grey')
plt.savefig('w_mean_std')

# Plot 3: H mean vs log N
plt.figure(3)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, H_mean[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('H mean (bits)')
plt.ylim(2.6,3.2)
plt.grid(axis='y', linestyle='--', color='grey')
plt.savefig('H_mean')

# Plot 4: H mean std vs log N
plt.figure(4)
x = range(num_powers) + np.ones(num_powers)
for i in range(num_trials):
    plt.scatter(x, H_mean_std[i,:], color='black')
plt.xlabel('log N')
plt.ylabel('H mean std (bits)')
plt.ylim(0.0,0.04)
plt.grid(axis='y', linestyle='--', color='grey')
plt.savefig('H_mean_std')

print ''
print '--- cal done ---'

df_w_mean = pd.DataFrame(data=w_mean, columns=['1','2','3','4','5','6'])
df_w_mean.index.name = '# trial'
df_w_mean.to_csv('w_mean.txt', sep='\t')

df_w_mean_std = pd.DataFrame(data=w_mean_std, columns=['1','2','3','4','5','6'])
df_w_mean_std.index.name = '# trial'
df_w_mean_std.to_csv('w_mean_std.txt', sep='\t')

df_H_mean = pd.DataFrame(data=H_mean, columns=['1','2','3','4','5','6'])
df_H_mean.index.name = '# trial'
df_H_mean.to_csv('H_mean.txt', sep='\t')

df_H_mean_std = pd.DataFrame(data=H_mean_std, columns=['1','2','3','4','5','6'])
df_H_mean_std.index.name = '# trial'
df_H_mean_std.to_csv('H_mean_std.txt', sep='\t')

print ''
print '--- df done ---'

sns.set(style='whitegrid')

plt.figure(5)
fig_w_mean = sns.violinplot(data=df_w_mean, color='white', inner='quart')
fig_w_mean.set_xlabel('log_N')
fig_w_mean.set_ylabel('w mean')
fig_w_mean.set_ylim(0.7, 1.3)
plt.savefig('w_mean_vp')

plt.figure(6)
fig_w_mean_std = sns.violinplot(data=df_w_mean_std, color='white')
fig_w_mean_std.set_xlabel('log_N')
fig_w_mean_std.set_ylabel('w mean std')
fig_w_mean_std.set_ylim(0.0, 0.1)
plt.savefig('w_mean_std_vp')

plt.figure(7)
fig_H_mean = sns.violinplot(data=df_H_mean, color='white', inner='quart')
fig_H_mean.set_xlabel('log_N')
fig_H_mean.set_ylabel('H mean (bits)')
plt.savefig('H_mean_vp')

plt.figure(8)
fig_H_mean_std = sns.violinplot(data=df_H_mean_std, color='white')
fig_H_mean_std.set_xlabel('log_N')
fig_H_mean_std.set_ylabel('H mean std (bits')
fig_H_mean_std.set_ylim(0.0, 0.01)
plt.savefig('H_mean_std_vp')

print ''
print '--- sns done ---'
print ''
