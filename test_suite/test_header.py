from __future__ import division
import scipy as sp
import numpy as np
import sys
import os

# Add parent directory to path
#sys.path.append('../code/')
sys.path.insert(0,os.getcwd()+'/sim')
sys.path.append('../sim/')

# Import deft modules
import simulate_data_1d
import deft_1d
import deft_core
import utils

# Import error handling
from utils import DeftError

# Counts the number of test cases
global_counter = 0


class TestCase:
    """ Contains information needed for a test run of deft_1d. """

    def __init__(self, message='', should_succeed=True, feed_data=False, data_fed=None, Q_true_func=None,
                 data_type='wide', N=100, data_seed=None, G=100, alpha=3, bbox=[-6,6],bbox_state=1,input_type='simulated', periodic=False,
                 Z_eval='Lap', num_Z_samples=0, DT_MAX=1.0, print_t=False, tollerance=1E-6, resolution=1E-1,
                 deft_seed=None, pt_method=None, num_pt_samples=0, fix_t_at_t_star=True):
        
        # Set internal variables
        self.message = message
        self.should_succeed = should_succeed
        self.feed_data = feed_data
        self.data_fed = data_fed
        self.Q_true_func = Q_true_func
        self.data_type = data_type
        self.N = N
        self.data_seed = data_seed        
        self.G = G
        self.alpha = alpha
        self.bbox = bbox
        self.bbox_state = bbox_state
        self.periodic = periodic
        self.Z_eval = Z_eval
        self.num_Z_samples = num_Z_samples
        self.DT_MAX = DT_MAX
        self.print_t = print_t
        self.tollerance = tollerance
        self.resolution = resolution
        self.deft_seed = deft_seed
        self.pt_method = pt_method
        self.num_pt_samples = num_pt_samples
        self.fix_t_at_t_star = fix_t_at_t_star
        self.input_type = input_type

        self.outcome_good = False

        # Increment test case number
        global global_counter
        self.test_num = global_counter
        global_counter += 1
        
    def run(self):

        # Use the data fed or generate data anew
        if self.feed_data:
            self.data = self.data_fed
            #print(self.data)
        else:
            self.data, self.details = simulate_data_1d.run(self.data_type, self.N, self.data_seed)

            # if auto bbox, then choose from simulate 1d
            if self.bbox_state==1 and self.input_type=='simulated':
                self.bbox = [self.details['box_min'],self.details['box_max']]
        # Check inputs
        inputs_check(data=self.data, G=self.G, alpha=self.alpha, bbox=self.bbox,
                     periodic=self.periodic, Z_eval=self.Z_eval, DT_MAX=self.DT_MAX,
                     print_t=self.print_t, tollerance=self.tollerance, resolution=self.resolution,
                     deft_seed=self.deft_seed, pt_method=self.pt_method,
                     fix_t_at_t_star=self.fix_t_at_t_star, num_pt_samples=self.num_pt_samples)

        # Run deft_1d
        try:
            self.results = deft_1d.run(data=self.data, G=self.G, alpha=self.alpha, bbox=self.bbox,
                                       periodic=self.periodic, Z_eval=self.Z_eval, num_Z_samples=self.num_Z_samples,
                                       DT_MAX=self.DT_MAX, print_t=self.print_t, tollerance=self.tollerance,
                                       resolution=self.resolution, deft_seed=self.deft_seed, pt_method=self.pt_method,
                                       num_pt_samples=self.num_pt_samples, fix_t_at_t_star=self.fix_t_at_t_star)
            print('Succeeded!  t_star = %.2f' % self.results.t_star)
            print(self.pt_method)
            self.outcome_good = self.should_succeed
        except DeftError as e:
            print(e)
            print('Failed in a controlled way.')
            self.outcome_good = not self.should_succeed
            self.ERROR_switch = True
        except:
            print('Problem! Test failed in an uncontrolled way!')
            self.outcome_good = False
            self.ERROR_switch = True
            
        if self.outcome_good:
            print('Outcome: good.')
            # Check numerics of the MAP curve
            numerics_check(self)
            return self
        else:
            print('Outcome: BAD!')
            print('')
            return self


# Check inputs
def inputs_check(data, G, alpha, bbox, periodic, Z_eval, DT_MAX, print_t, tollerance,
                 resolution, deft_seed, pt_method, fix_t_at_t_star, num_pt_samples):

    # Make sure data is valid
    if not isinstance(data, utils.ARRAY):
        raise DeftError('/inputs/ data must be an array: data = %s' % type(data))
    if not (len(data) > 0):
        raise DeftError('/inputs/ data must have length > 0: data = %s' % data)
    for i in range(len(data)):
        if isinstance(data[i], bool):
            raise DeftError('/inputs/ data must contain numbers: data = %s' % data)
        if not isinstance(data[i], utils.NUMBER):
            raise DeftError('/inputs/ data must contain numbers: data = %s' % data)
    data_spread = max(data) - min(data)
    if not np.isfinite(data_spread):
        raise DeftError('/inputs/ data_spread is not finite: data_spread = %s' % data_spread)
    if not (data_spread > 0):
        raise DeftError('/inputs/ data_spread is not > 0: data_spread = %s' % data_spread)

    # Make sure G is valid
    if isinstance(G, bool):
        raise DeftError('/inputs/ G must be an integer: G = %s' % type(G))
    if not isinstance(G, int):
        raise DeftError('/inputs/ G must be an integer: G = %s' % type(G))
    if not (G >= 50):
        raise DeftError('/inputs/ G must be >= 50: G = %s' % G)

    # Make sure alpha is valid
    if isinstance(alpha, bool):
        raise DeftError('/inputs/ alpha must be an integer: alpha = %s' % type(alpha))
    if not isinstance(alpha, int):
        raise DeftError('/inputs/ alpha must be an integer: alpha = %s' % type(alpha))
    if not ((alpha >= 1) and (alpha <= 5)):
        raise DeftError('/inputs/ alpha must be 1 <= alpha <= 5: alpha = %s' % alpha)

    # Make sure bbox is valid
    if not isinstance(bbox, utils.ARRAY):
        raise DeftError('/inputs/ bbox must be an array: bbox = %s' % type(bbox))
    if not (len(bbox) == 2):
        raise DeftError('/inputs/ bbox must have length 2: bbox = %s' % bbox)
    for i in range(2):
        if isinstance(bbox[i], bool):
            raise DeftError('/inputs/ bbox must contain numbers: bbox = %s' % bbox)
        if not isinstance(bbox[i], utils.NUMBER):
            raise DeftError('/inputs/ bbox must contain numbers: bbox = %s' % bbox)
    if not (bbox[0] < bbox[1]):
        raise DeftError('/inputs/ bbox[1] is not > bbox[0]: bbox = %s' % bbox)

    # Make sure periodic is valid
    if not isinstance(periodic, bool):
        raise DeftError('/inputs/ periodic must be a boolean: periodic = %s' % type(periodic))

    # Make sure Z_eval is valid
    Z_evals = ['Lap', 'Lap+Sam', 'Lap+Sam+P', 'GLap', 'GLap+P', 'GLap+Sam', 'GLap+Sam+P', 'Lap+Fey']
    if not (Z_eval in Z_evals):
        raise DeftError('/inputs/ Z_eval must be in %s: Z_eval = %s' % (Z_evals,Z_eval))

    # Make sure DT_MAX is valid
    if not isinstance(DT_MAX, float):
        raise DeftError('/inputs/ DT_MAX must be a float: DT_MAX = %s' % type(DT_MAX))
    if not (DT_MAX > 0):
        raise DeftError('/inputs/ DT_MAX must be > 0: DT_MAX = %s' % DT_MAX)

    # Make sure print_t is valid
    if not isinstance(print_t, bool):
        raise DeftError('/inputs/ print_t must be a boolean: print_t = %s' % type(print_t))

    # Make sure tollerance is valid
    if not isinstance(tollerance, float):
        raise DeftError('/inputs/ tollerance must be a float: tollerance = %s' % type(tollerance))
    if not (tollerance > 0):
        raise DeftError('/inputs/ tollerance must be > 0: tollerance = %s' % tollerance)

    # Make sure resolution is valid
    if not isinstance(resolution, float):
        raise DeftError('/inputs/ resolution must be a float: resolution = %s' % type(resolution))
    if not (resolution > 0):
        raise DeftError('/inputs/ resolution must be > 0: resolution = %s' % resolution)

    # Make sure deft_seed is valid
    if isinstance(deft_seed, bool):
        raise DeftError('/inputs/ deft_seed must be either None or an integer: deft_seed = %s' % type(deft_seed))
    if not ((deft_seed is None) or isinstance(deft_seed, int)):
        raise DeftError('/inputs/ deft_seed must be either None or an integer: deft_seed = %s' % type(deft_seed))
    if (deft_seed is not None) and ((deft_seed < 0) or (deft_seed > 2**32-1)):
        raise DeftError('/inputs/ deft_seed must be 0 <= deft_seed <= 2**32-1: deft_seed = %s' % deft_seed)

    # Make sure pt_method is valid
    pt_methods = [None, 'Lap', 'Lap+P', 'Lap+W', 'Lap+W+P', 'GLap', 'GLap+P', 'GLap+W', 'GLap+W+P', 'MMC']
    if not (pt_method in pt_methods):
        raise DeftError('/inputs/ pt_method must be in %s: pt_method = %s' % (pt_methods,pt_method))

    # Make sure fix_t_at_t_star is valid
    if not isinstance(fix_t_at_t_star, bool):
        raise DeftError('/inputs/ fix_t_at_t_star must be a boolean: fix_t_at_t_star = %s' % type(fix_t_at_t_star))

    # Make sure num_pt_samples is valid
    if isinstance(num_pt_samples, bool):
        raise DeftError('/inputs/ num_pt_samples must be an integer: num_pt_samples = %s' % type(num_pt_samples))
    if not isinstance(num_pt_samples, int):
        raise DeftError('/inputs/ num_pt_samples must be an integer: num_pt_samples = %s' % type(num_pt_samples))
    if not (num_pt_samples >= 0):
        raise DeftError('/inputs/ num_pt_samples must be >= 0: num_pt_samples = %s' % num_pt_samples)


# Check numerics of the MAP curve
def numerics_check(self):
    ERROR_switch = False
    print('------')
    # Read in t and Q (properly normalized)
    ts = sp.array([p.t for p in self.results.points])
    Qs = sp.array([p.Q for p in self.results.points])
    num_t = len(ts)

    # Other stuff needed
    G = self.results.G
    N = self.results.N
    Delta = self.results.Delta
    alpha = self.results.alpha
    h = self.results.h
    bbox_lb = self.results.bbox[0]
    bbox_ub = self.results.bbox[1]
    resolution = self.results.resolution
    break_t_loop = self.results.map_curve.break_t_loop

    # (1) Check actions
    actions = np.zeros([num_t,3])
        
    # Compute Q_true and phi_true
    xs = self.results.bin_centers
    if self.feed_data:
        # these tests shouldn't be performed if data is fed
        pass
        #Q_true_func = self.Q_true_func
        #Q_true = Q_true_func(xs)
        #Q_true = Q_true/sp.sum(Q_true)
        #phi_true = utils.prob_to_field(Q_true)
    else:
        Q_true = np.zeros(G)
        for i, x in enumerate(xs):
            Q_true[i] = eval(self.details['pdf_py'])
        Q_true = Q_true/sp.sum(Q_true)
        phi_true = utils.prob_to_field(Q_true)
        self.Q_true = Q_true/h

    # For each t, compute action of phi and phi_true, and then compare
    R = Qs[-1]/sp.sum(Qs[-1])
    for i in range(1,num_t): # excluding maxent: S[phi_true] at maxent is infinity
        t_i = ts[i]
        Q_i = Qs[i]/sp.sum(Qs[i])
        phi_i = utils.prob_to_field(Q_i)
        S_of_phi_i = deft_core.action(phi_i, R, Delta, t_i, N) * (N/G)
        #S_of_phi_true = deft_core.action(phi_true, R, Delta, t_i, N) * (N/G)
        actions[i,0] = S_of_phi_i
        #actions[i,1] = S_of_phi_true
        #actions[i,2] = S_of_phi_i - S_of_phi_true

    # Error checking
    if not all(actions[:,2] <= 0.0):
        print('ERROR: actions')
        ERROR_switch = True

    # (2) Check moments
    moments = np.zeros([num_t,alpha])

    # Shift xs so that all xs > 0 to avoid divergence in odd-moments
    if (bbox_lb < 0) and (bbox_ub > 0):
        xs_shift = np.copy(xs) - bbox_lb
    else:
        xs_shift = np.copy(xs)

    # k-th power of xs
    xs_shift_k = np.zeros([G,alpha])
    for k in range(alpha):
        xs_shift_k[:,k] = xs_shift**k

    # For each t, compute moments of Q up to order alpha-1
    for i in range(num_t):
        Q_i = Qs[i]
        for k in range(alpha):
            x_k = xs_shift_k[:,k]
            moments[i,k] = sp.sum(Q_i*x_k) * h
            
    # Compute percent difference of k-moments w.r.t. that of R
    for k in range(alpha):
        moments[:,k] = (moments[:,k]/moments[-1,k]-1.0) * 100.0

    # Error checking
    for k in range(alpha):
        if not all(abs(moments[:,k]) <= 1.0):
            print('ERROR: %s-moments: %.2f %%' % (k,max(abs(moments[:,k]))))
            ERROR_switch = True

    # (3) Check entropies
    entropies = np.zeros([num_t,3])
        
    # Compute entropy of maxent
    M = Qs[0]
    H_M = -sp.sum(M*sp.log2(M + utils.TINY_FLOAT64)) * h
        
    # For each t, compute entropy of Q and then compare to that of maxent
    for i in range(num_t):
        Q_i = Qs[i]
        H_i = -sp.sum(Q_i*sp.log2(Q_i + utils.TINY_FLOAT64)) * h
        entropies[i,0] = H_i
        entropies[i,1] = H_M
        entropies[i,2] = H_i - H_M
        
    # Error checking
    if not all(entropies[:,2] <= 0.0):
        print('ERROR: entropies')
        ERROR_switch = True

    # (4) Check geo_dists
    geo_dists = np.zeros(num_t-1)

    # For each pair of t, compute geo_distance between them and divide by resolution
    for i in range(num_t-1):
        Q_i = Qs[i]
        Q_j = Qs[i+1]
        geo_dist = utils.geo_dist(Q_i,Q_j)
        geo_dists[i] = geo_dist/resolution
    
    # Error checking
    if break_t_loop == [True,True]:
        if not all(geo_dists <= 1.0): 
            print('ERROR: geo_dists')
            ERROR_switch = True
    if break_t_loop == [True,False]: # excluding (last Q, R)
        if not all(geo_dists[:-1] <= 1.0): 
            print('ERROR: geo_dists')
            ERROR_switch = True
    if break_t_loop == [False,True]: # excluding (first Q, M)
        if not all(geo_dists[1:] <= 1.0): 
            print('ERROR: geo_dists')
            ERROR_switch = True
    if break_t_loop == [False,False]: # excluding both (first Q, M), and (last Q, R)
        if not all(geo_dists[1:-1] <= 1.0):
            print('ERROR: geo_dists')
            ERROR_switch = True

    # Return error indicator
    self.ERROR_switch = ERROR_switch
