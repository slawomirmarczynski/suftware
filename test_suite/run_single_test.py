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
import matplotlib.pyplot as plt

execfile('test_header.py')

results = TestCase(N=100, G=100,
                   data_seed=0,
                   DT_MAX=1.0,
                   alpha=3,
                   Z_eval='GLap',
                   pt_method=None,
                   fix_t_at_t_star=True,
                   num_pt_samples=100,
                   bbox=[-6,6]
                   ).run()

# Print out effective sample size at each t
ts = [p.t for p in results.results.points]
ESSs = [p.eff_sample_size for p in results.results.points]
print('')
print('Partition function evaluation: %s' % results.Z_eval)
for i in range(len(ts)):
    if isinstance(ESSs[i], float):
        print('Effective sample size = %.2f at t = %.2f' % (ESSs[i],ts[i]))
    else:
        print('Effective sample size = %s at t = %.2f' % (ESSs[i], ts[i]))
print('')

# Print out effective sample size and make a plot of Q's
if not (results.pt_method is None):
    print('Posterior sampling: %s' % results.pt_method)
    print('Number of samples     = %s' % results.results.num_pt_samples)
    print('Effective sample size = %s' % results.results.pt_eff_sample_size)
    print('')
    xs = results.results.bin_centers
    Q_star = results.results.Q_star
    Q_samples = results.results.Q_samples
    plt.plot(xs, Q_samples, color='blue', alpha=0.3)
    plt.plot(xs, Q_star, color='red')
    plt.ylim([0, max(Q_star)*1.5])
    plt.show()
