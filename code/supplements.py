from __future__ import division
import numpy as np
import scipy as sp
from scipy.sparse import diags
import multiprocessing as mp
import itertools
import time
import sys

if sys.version_info[0]==2:
    # python 2 imports
    import deft_core
    import utils
    import maxent
    from deft_code.utils import DeftError

else:
    # python 3 imports
    from deft_code import deft_core
    from deft_code import maxent
    from deft_code import utils
    from deft_code.utils import DeftError

x_MIN = -500


# Laplace approach with importance sampling
def Laplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel, pt_sampling=False):

    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N/G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G,alpha])
        for i in range(alpha):
            kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N/G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = sp.array(U_mat_on_kernel[0])
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N/G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        #H = deft_code.deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N/G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = sp.array(U_mat[0])
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), eig_vals)
        outputs = pool.map(y_sampling_of_Lap, inputs)
        y_samples = sp.array(outputs)
    else:
        y_samples = np.zeros([y_dim,num_samples])
        for i in range(y_dim):
            inputs = [num_samples, eig_vals[i]]
            outputs = y_sampling_of_Lap(inputs)
            y_samples[i,:] = outputs

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i, :]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G,num_samples])
    for k in range(num_samples):
        phi_samples[:,k] = x_samples[:,k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones([G,num_samples]) + x_samples - 0.5*np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    phi_weights = sp.exp(-dS_vals)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean and sample mean std
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    correction = sp.log(w_sample_mean)
    return correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, draw y samples according to the distribution
def y_sampling_of_Lap(input_array):

    num_samples = input_array[0]
    eig_val = input_array[1]

    # Draw y samples
    sigma = 1.0/sp.sqrt(eig_val)
    y_samples = np.random.normal(0, sigma, num_samples)

    # Return y samples
    return y_samples


# Generalized Laplace approach with importance sampling
def GLaplace_approach(phi_t, R, Delta, t, N, num_samples, go_parallel, sampling=True, pt_sampling=False, num_grid=400):

    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N/G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G,alpha])
        for i in range(alpha):
            kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N/G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
        # Below are what will be used
        y_dim = alpha
        eig_vals = sp.array(U_mat_on_kernel[0])
        eig_vecs = sp.array((sp.mat(kernel_basis) * U_mat_on_kernel[1]).T)
        transf_matrix = sp.mat(kernel_basis) * U_mat_on_kernel[1]
        lambdas = sp.exp(-phi_t) * (N/G)
    else:
        G = len(phi_t)
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N/G)
        U_mat = np.linalg.eigh(A_mat)
        # Below are what will be used
        y_dim = G
        eig_vals = sp.array(U_mat[0])
        eig_vecs = sp.array(U_mat[1].T)
        transf_matrix = U_mat[1]
        lambdas = sp.exp(-phi_t) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, calculate gamma and draw y samples according to the distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), itertools.repeat(num_grid),
                                eig_vals, eig_vecs, itertools.repeat(lambdas), itertools.repeat(sampling))
        outputs = pool.map(y_sampling_of_GLap, inputs)
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim,num_samples])
        for i in range(y_dim):
            gammas[i] = outputs[i][0]
            if sampling:
                y_samples[i,:] = outputs[i][1]
    else:
        gammas = np.zeros(y_dim)
        y_samples = np.zeros([y_dim,num_samples])
        for i in range(y_dim):
            inputs = [num_samples, num_grid, eig_vals[i], eig_vecs[i,:], lambdas, sampling]
            outputs = y_sampling_of_GLap(inputs)
            gammas[i] = outputs[0]
            if sampling:
                y_samples[i,:] = outputs[1]

    # If not sampling, return correction and other stuff at this point
    if not sampling:
        correction = sp.sum(sp.log(gammas))
        w_sample_mean = 1.0
        w_sample_mean_std = 0.0
        return correction, w_sample_mean, w_sample_mean_std

    # Transform y samples to x samples
    x_samples = sp.array(transf_matrix * sp.mat(y_samples))
    for i in range(G):
        x_vec = x_samples[i,:]
        x_vec[x_vec < x_MIN] = x_MIN

    # Shift x samples to get phi samples
    phi_samples = np.zeros([G,num_samples])
    for k in range(num_samples):
        phi_samples[:,k] = x_samples[:,k] + phi_t

    # Calculate the weight of each sample
    x_combo = sp.exp(-x_samples) - np.ones([G,num_samples]) + x_samples - 0.5*np.square(x_samples)
    dS_vals = sp.array(sp.mat(lambdas) * sp.mat(x_combo)).ravel()
    if go_parallel:
        inputs = itertools.izip(sp.array(transf_matrix.T), y_samples, itertools.repeat(lambdas))
        outputs = pool.map(dSi_evaluations_of_GLap, inputs)
        dSi_vals = sp.array(outputs)
    else:
        dSi_vals = np.zeros([y_dim,num_samples])
        for i in range(y_dim):
            inputs = [sp.array(transf_matrix)[:,i], y_samples[i,:], lambdas]
            outputs = dSi_evaluations_of_GLap(inputs)
            dSi_vals[i,:] = outputs
    sum_dSi_vals = sp.array(sp.mat(np.ones(y_dim)) * sp.mat(dSi_vals)).ravel()
    dS_residues = dS_vals - sum_dSi_vals
    dS_residues[dS_residues < x_MIN] = x_MIN
    phi_weights = sp.exp(-dS_residues)

    # If called from posterior sampling, return phi samples along with their weights at this point
    if pt_sampling:
        return phi_samples, phi_weights

    # Calculate sample mean, sample mean std, and effective sample size of the weights
    w_sample_mean = sp.mean(phi_weights)
    w_sample_mean_std = sp.std(phi_weights) / sp.sqrt(num_samples)

    # Return correction and other stuff
    correction = sp.sum(sp.log(gammas)) + sp.log(w_sample_mean)
    return correction, w_sample_mean, w_sample_mean_std


# For each eigen-component, calculate gamma and draw y samples according to the distribution
def y_sampling_of_GLap(inputs_array):

    num_samples = inputs_array[0]
    num_grid = inputs_array[1]
    eig_val = inputs_array[2]
    eig_vec = inputs_array[3]
    lambdas = inputs_array[4]
    sampling = inputs_array[5]

    # Find the lower and upper bounds of the Laplace distribution and tabulate its values
    sigma = 1.0/sp.sqrt(eig_val)
    Lap_N_lb = 0
    while distribution(eig_val, eig_vec, Lap_N_lb*sigma, lambdas, switch=0) > 1E-6:
        Lap_N_lb -= 1
    Lap_N_ub = 0
    while distribution(eig_val, eig_vec, Lap_N_ub*sigma, lambdas, switch=0) > 1E-6:
        Lap_N_ub += 1
    Lap_Ns = []
    Lap_Es = []
    for Lap_N in range(Lap_N_lb, Lap_N_ub+1):
        Lap_Ns.append(Lap_N)
        Lap_Es.append(distribution(eig_val, eig_vec, Lap_N*sigma, lambdas, switch=0))

    # Find the lower and upper bounds of the generalized Laplace distribution and tabulate its values
    sigma = 1.0/sp.sqrt(eig_val)
    GLap_N_lb = 0
    while distribution(eig_val, eig_vec, GLap_N_lb*sigma, lambdas, switch=1) > 1E-6:
        GLap_N_lb -= 1
    GLap_N_ub = 0
    while distribution(eig_val, eig_vec, GLap_N_ub*sigma, lambdas, switch=1) > 1E-6:
        GLap_N_ub += 1
    GLap_Ns = []
    GLap_Es = []
    for GLap_N in range(GLap_N_lb, GLap_N_ub+1):
        GLap_Ns.append(GLap_N)
        GLap_Es.append(distribution(eig_val, eig_vec, GLap_N*sigma, lambdas, switch=1))

    # See if these two distributions are similar enough
    if Lap_Ns == GLap_Ns:
        diff_Es = abs(sp.array(Lap_Es) - sp.array(GLap_Es))
        if all(diff_Es < 1E-6):
            similar_enough = True
        else:
            similar_enough = False
    else:
        similar_enough = False

    # If these two distributions are similar enough, set gamma to 1, otherwise do the integration
    if similar_enough:
        gamma = 1.0
    else:
        # Evaluate area under the Laplace distribution
        Lap_bin_edges = sp.linspace(Lap_Ns[0]*sigma, Lap_Ns[-1]*sigma, num_grid+1)
        h = Lap_bin_edges[1] - Lap_bin_edges[0]
        Lap_bin_centers = sp.linspace(Lap_Ns[0]*sigma+h/2, Lap_Ns[-1]*sigma-h/2, num_grid)
        Lap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            Lap_bin_centers_dist[j] = distribution(eig_val, eig_vec, Lap_bin_centers[j], lambdas, switch=0)
        Lap_area = h * sp.sum(Lap_bin_centers_dist)
        # Evaluate area under the generalized Laplace distribution
        GLap_bin_edges = sp.linspace(GLap_Ns[0]*sigma, GLap_Ns[-1]*sigma, num_grid+1)
        h = GLap_bin_edges[1] - GLap_bin_edges[0]
        GLap_bin_centers = sp.linspace(GLap_Ns[0]*sigma+h/2, GLap_Ns[-1]*sigma-h/2, num_grid)
        GLap_bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            GLap_bin_centers_dist[j] = distribution(eig_val, eig_vec, GLap_bin_centers[j], lambdas, switch=1)
        GLap_area = h * sp.sum(GLap_bin_centers_dist)
        # Take ratio of the areas
        gamma = GLap_area / Lap_area

    # If not sampling, return gamma at this point
    if not sampling:
        return [gamma]

    # If the distribution is normal, draw samples from it directly
    if similar_enough:
        y_samples = np.random.normal(0, sigma, num_samples)

    # Otherwise, draw samples according to the distribution as follows
    else:
        bin_edges = sp.linspace(GLap_N_lb*sigma, GLap_N_ub*sigma, num_grid+1)
        h = bin_edges[1] - bin_edges[0]
        bin_centers = sp.linspace(GLap_N_lb*sigma+h/2, GLap_N_ub*sigma-h/2, num_grid)
        bin_centers_dist = np.zeros(num_grid)
        for j in range(num_grid):
            bin_centers_dist[j] = distribution(eig_val, eig_vec, bin_centers[j], lambdas, switch=1)
        prob = bin_centers_dist / sp.sum(bin_centers_dist)
        y_samples = np.random.choice(bin_centers, num_samples, replace=True, p=prob)
        y_shifts = (np.random.random(num_samples) - 0.5*np.ones(num_samples)) * h
        y_samples += y_shifts
        """
            Below is a HOT spot !!!

            # Randomly distribute the samples within each bin
            indices = (y_samples-h/2-N_lb*sigma) / h
            for k in range(num_samples):
                index = int(indices[k])
                a = y_grid[index]
                fa = grid_dist[index]
                fb = grid_dist[index+1]
                r = np.random.rand()
                if fa == fb:
                    y_samples[k] = a + h * r
                else:
                    h_ratio = (sp.sqrt(fa**2+r*(fb**2-fa**2)) - fa) / (fb - fa)
                    y_samples[k] = a + h * h_ratio
            """

    # Return gamma and y samples
    return [gamma, y_samples]


# Evaluations of dSi
def dSi_evaluations_of_GLap(inputs_array):

    Ui = inputs_array[0]
    yi = inputs_array[1]
    lambdas = inputs_array[2]

    G = len(Ui)
    num_samples = len(yi)

    xi = sp.array(sp.mat(Ui).T * sp.mat(yi))
    for i in range(G):
        xi_vec = xi[i,:]
        xi_vec[xi_vec < x_MIN] = x_MIN
    xi_combo = sp.exp(-xi) - np.ones([G,num_samples]) + xi - 0.5*np.square(xi)

    return sp.array(sp.mat(lambdas) * sp.mat(xi_combo)).ravel()


# The Laplace or generalized Laplace distribution
def distribution(eig_val, eig_vec, y, lambdas, switch):
    return sp.exp(-(0.5*eig_val*y**2 + switch * dSi(eig_vec*y, lambdas)))


# The dSi function
def dSi(x, lambdas):
    x[x < x_MIN] = x_MIN
    return sp.sum(lambdas * (sp.exp(-x) - 1.0 + x - 0.5*x**2))


# Feynman diagram calculations
def Feynman_diagrams(phi_t, R, Delta, t, N):

    # Prepare the stuff for the case of maxent or finite t
    if not np.isfinite(t):
        G = len(phi_t)
        alpha = Delta._kernel_dim
        # Evaluate propagator matrix
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N/G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G,alpha])
        for i in range(alpha):
            kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
        M_mat = diags(sp.exp(-phi_t), 0).todense() * (N/G)
        M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
        M_inv_on_kernel = sp.linalg.inv(M_mat_on_kernel)
        P_mat = sp.mat(kernel_basis) * M_inv_on_kernel * sp.mat(kernel_basis).T
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N/G)
    else:
        G = len(phi_t)
        # Evaluate propagator matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N/G)
        P_mat = np.linalg.inv(A_mat)
        # Evaluate vertex vector
        V = sp.exp(-phi_t) * (N/G)

    # Calculate Feynman diagrams
    correction = diagrams_1st_order(G, P_mat, V)

    # Return the correction and other stuff
    w_sample_mean = 1.0
    w_sample_mean_std = 0.0
    return correction, w_sample_mean, w_sample_mean_std


# Feynman diagrams of order 1/N
def diagrams_1st_order(G, P, V):
    s = np.zeros(4) # s[0] is dummy

    P_diag = sp.array([P[i,i] for i in range(G)])

    # Diagram 1
    s[1] = sp.sum(V * P_diag**2)
    s[1] *= -1/8

    # Diagram 2
    U = sp.array([V[i]*P_diag[i] for i in range(G)])
    s[2] = sp.array(sp.mat(U) * P * sp.mat(U).T).ravel()[0]
    s[2] *= 1/8

    # Diagram 3
    s[3] = sp.array(sp.mat(V) * sp.mat(sp.array(P)**3) * sp.mat(V).T).ravel()[0]
    s[3] *= 1/12

    """
    # Diagram 1
    for i in range(G):
        s[1] += V[i] * P[i,i]**2
    s[1] *= -1/8

    # Diagram 2
    for i in range(G):
        for j in range(G):
            s[2] += V[i] * V[j] * P[i,i] * P[i,j] * P[j,j]
    s[2] *= 1/8

    # Diagram 3
    for i in range(G):
        for j in range(G):
            s[3] += V[i] * V[j] * P[i,j]**3
    s[3] *= 1/12
    """

    # Return
    return sp.sum(s)


# Feynman diagrams of order 1/N^2 ---> Under construction
def diagrams_2nd_order(G, P, V):


    time1 = time.time()
    num_samples = 300000 # Use different value for large and small t ?
    index_samples = np.random.randint(0, G-1, 4*num_samples).reshape([num_samples,4])

    s_array = np.zeros(num_samples)
    for n in range(num_samples):
        i = index_samples[n,0]
        j = index_samples[n,1]
        k = index_samples[n,2]
        l = index_samples[n,3]
        s_array[n] = V[i] * V[j] * V[k] * V[l] * P[i,j] * P[i,k] * P[i,l] * P[j,k] * P[j,l] * P[k,l]
    s = sp.sum(s_array) * G**4 / num_samples
    ms = sp.mean(s_array)
    ds = sp.std(s_array) / sp.sqrt(num_samples)
    print('')
    print('s =', s)
    print(ds/ms)
    print('time 1 =', time.time()-time1)

    """
    time2 = time.time()
    s2 = 0
    for i in range(G):
        for j in range(G):
            for k in range(G):
                for l in range(G):
                    s2 += V[i] * V[j] * V[k] * V[l] * P[i,j] * P[i,k] * P[i,l] * P[j,k] * P[j,l] * P[k,l]
    print 's2 =', s2
    print 'time 2 =', time.time()-time2
    """

    return 0


# Metropolis Monte Carlo
def Metropolis_Monte_Carlo(phi_t, R, Delta, t, N, num_samples, go_parallel, pt_sampling):

    G = len(phi_t)
    num_thermalization_steps = 10*G
    num_steps_per_sample = G

    phi_samples = np.zeros([G,num_samples])
    sample_index = 0

    # Prepare the stuff for the case of maxent or finite t, and then do Monte Carlo sampling
    if not np.isfinite(t):

        # Find the kernel basis
        alpha = Delta._kernel_dim
        Delta_sparse = Delta.get_sparse_matrix()
        Delta_mat = Delta_sparse.todense() * (N/G)
        Delta_diagonalized = np.linalg.eigh(Delta_mat)
        kernel_basis = np.zeros([G,alpha])
        for i in range(alpha):
            kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()

        # Find coefficients of phi_t in the kernel basis
        coeffs = np.zeros(alpha)
        for i in range(alpha):
            coeffs[i] = sp.mat(kernel_basis[:,i]) * sp.mat(phi_t).T

        # Find eigen-modes of the Hessian matrix
        H = maxent.hessian_per_datum_from_coeffs(coeffs, R, kernel_basis)
        A_mat = sp.mat(H) * N
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = sp.array(U_mat[0])
        eig_vecs = sp.array(U_mat[1])

        # Initialize
        coeffs_current = coeffs
        S_current = maxent.action_per_datum_from_coeffs(coeffs_current, R, kernel_basis) * N

        # Do Monte Carlo sampling
        for k in range(num_thermalization_steps + num_samples*num_steps_per_sample + 1):
            i = np.random.randint(0,alpha)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[i,:]
            step_size = np.random.normal(0, 1.0/np.sqrt(eig_val))
            coeffs_new = coeffs_current + eig_vec * step_size
            S_new = maxent.action_per_datum_from_coeffs(coeffs_new, R, kernel_basis) * N
            if np.log(np.random.uniform(0,1)) < (S_current-S_new):
                coeffs_current = coeffs_new
                S_current = S_new
            if (k > num_thermalization_steps) and (k%num_steps_per_sample == 0):
                phi_samples[:,sample_index] = maxent.coeffs_to_field(coeffs_current, kernel_basis)
                sample_index += 1

    else:

        # Find eigen-modes of the Hessian matrix
        H = deft_core.hessian(phi_t, R, Delta, t, N)
        A_mat = H.todense() * (N/G)
        U_mat = np.linalg.eigh(A_mat)
        eig_vals = sp.array(U_mat[0])
        eig_vecs = sp.array(U_mat[1])

        # Initialize
        phi_current = phi_t
        S_current = deft_core.action(phi_current, R, Delta, t, N) * (N/G)

        # Do Monte Carlo sampling
        for k in range(num_thermalization_steps + num_samples*num_steps_per_sample + 1):
            i = np.random.randint(0,G)
            eig_val = eig_vals[i]
            eig_vec = eig_vecs[:,i]
            step_size = np.random.normal(0,1.0/np.sqrt(eig_val))
            phi_new = phi_current + eig_vec * step_size
            S_new = deft_core.action(phi_new, R, Delta, t, N) * (N/G)
            if np.log(np.random.uniform(0,1)) < (S_current - S_new):
                phi_current = phi_new
                S_current = S_new
            if (k > num_thermalization_steps) and (k%num_steps_per_sample == 0):
                phi_samples[:,sample_index] = phi_current
                sample_index += 1

    # Return phi samples and phi weights
    return phi_samples, np.ones(num_samples)


# Sample probable densities using posterior probability
def posterior_sampling(points, R, Delta, N, G, pt_method, num_pt_samples, fix_t_at_t_star):

    if pt_method == 'Lap' or 'Lap+Imp':
        method, go_parallel = Laplace_approach, False
    if pt_method == 'Lap+P':
        method, go_parallel = Laplace_approach, True
    if pt_method == 'GLap':
        method, go_parallel = GLaplace_approach, False
    if pt_method == 'GLap+P':
        method, go_parallel = GLaplace_approach, True
    if pt_method == 'MMC':
        method, go_parallel = Metropolis_Monte_Carlo, False

    phi_samples = np.zeros([G,num_pt_samples])
    phi_weights = np.zeros(num_pt_samples)
    sample_index = 0

    # Read in t, phi, log_E, and w_sample_mean from MAP curve points
    ts = sp.array([p.t for p in points])
    phis = sp.array([p.phi for p in points])
    log_Es = sp.array([p.log_E for p in points])
    w_sample_means = sp.array([p.sample_mean for p in points])

    # Generate a "histogram" of t according to their relative probability
    num_t = len(ts)
    if fix_t_at_t_star:
        hist_t = np.zeros(num_t)
        hist_t[log_Es.argmax()] = num_pt_samples
    else:
        log_Es = log_Es - log_Es.max()
        prob_t = sp.exp(log_Es)
        prob_t = prob_t / sp.sum(prob_t)
        num_indices = num_t
        sampled_indices = list(np.random.choice(num_indices, size=num_pt_samples, replace=True, p=prob_t))
        hist_t = [sampled_indices.count(c) for c in range(num_indices)]

    # Traverse through t, and draw a number of phi samples for each t
    for i in range(num_t):
        num_samples = int(hist_t[i])
        if num_samples > 0:
            t = ts[i]
            phi_t = phis[i]
            phi_samples_at_t, phi_weights_at_t = \
                method(phi_t, R, Delta, t, N, num_samples, go_parallel, pt_sampling=True)
            for k in range(num_samples):
                phi_samples[:,sample_index] = phi_samples_at_t[:,k]
                phi_weights[sample_index] = phi_weights_at_t[k] / w_sample_means[i]
                sample_index += 1

    # Convert phi samples to Q samples
    Q_samples = np.zeros([G,num_pt_samples])
    for k in range(num_pt_samples):
        Q_samples[:,k] = utils.field_to_prob(sp.array(phi_samples[:,k]).ravel())

    # Return Q samples along with their weights
    return Q_samples, phi_samples, phi_weights


# This method will be used to clean user input data; it's for use with the API.
def clean_data(data):

    try:
        # if data is a list or set, cast into numpy array
        if type(data) == list or type(data) == set:
            data = np.array(data)
        # if data already np array, do nothing
        elif type(data) == np.ndarray:
            pass
        # think about whether the following is a good idea
        elif type(data) != np.ndarray:
            data = np.array(data)
        else:
            raise DeftError("Error: could not cast data into an np.array")
    except DeftError as e:
        print(e)
        sys.exit(1)

    # remove nan's from the np data array
    data = data[~np.isnan(data)]
    # remove positive or negative infinite values from the np data array
    data = data[~np.isinf(data)]
    # remove complex numbers from data
    data = data[~np.iscomplex(data)]

    try:
        if not (len(data) > 0):
            raise DeftError('Input check failed, data must have length > 0: data = %s' % data)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # ensure data are numbers
    try:
        for i in range(len(data)):
            if not isinstance(data[i], utils.NUMBER):
                raise DeftError('Input check failed. data must contain numbers: data = %s' % data)
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        data_spread = max(data) - min(data)
        if not np.isfinite(data_spread):
            raise DeftError('Input check failed. Data[max]-Data[min] is not finite: Data spread = %s' % data_spread)
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (data_spread > 0):
            raise DeftError('Input check failed. Data[max]-Data[min] must be > 0: data_spread = %s' % data_spread)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # min difference between sorted, unique, adjacent values
    # should h be smaller than this value?

    min_h = np.min(np.diff(np.sort(np.unique(data))))

    # return cleaned data
    return data, min_h


# Check inputs
def inputs_check(G, alpha, bbox, periodic, Z_eval, DT_MAX, print_t, tollerance,
                 resolution, deft_seed, pt_method, fix_t_at_t_star, num_pt_samples,max_log_evidence_ratio_drop):

    # Make sure alpha is valid
    try:
        if not isinstance(alpha, int):
            raise DeftError('Input check failed. Parameter "alpha" must be an integer: alpha = %s' % type(alpha))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure G is valid
    try:
        if not isinstance(G, int):
            raise DeftError('Input check failed. Parameter "num_grid_points" must be an integer: num_grid_points = %s' % type(G))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (G >= 2*alpha and G <= 1000):
            raise DeftError('Input check failed. Parameter "num_grid_points" must between [2*alpha, 1000]: num_grid_points = %s' % G)
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        # the following values of alpha are used in the paper
        if not ((alpha >= 1) and (alpha <= 4)):
            raise DeftError('Input check failed. Parameter "alpha" must be 1 <= alpha <= 4: alpha = %s' % alpha)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure bbox is valid
    try:
        if not (bbox=='Auto' or isinstance(bbox,tuple)):
            raise TypeError

    except TypeError as e:
        print('Input check failed:  bounding box must be "Auto" or a 2-tuple of floats \nCurrent bounding box type = %s' % type(bbox))
        sys.exit(1)

    try:
        if bbox!='Auto':
            if not (len(bbox) == 2):
                print('bbox: ',bbox)
                raise DeftError('Input check failed. Bounding box must have length 2. Current length = %d' %len(bbox))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if bbox != 'Auto':
            for i in range(2):
                if not isinstance(bbox[i], utils.NUMBER):
                    raise DeftError('Input check failed. Bounding box must contain numbers: Bounding box entered = %s' % bbox)
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if bbox != 'Auto':
            if not (bbox[0] < bbox[1]):
                raise DeftError('Inputs check failed. bounding-box[1] should be greater than bounding-box[0].')
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure periodic is valid
    try:
        if not isinstance(periodic, bool):
            raise DeftError('Input check failed. Parameter "periodic" must be of type boolean: periodic = %s' % type(periodic))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure Z_eval is valid
    Z_evals = ['Lap', 'Lap+Imp', 'Lap+Fey']
    try:
        if not (Z_eval in Z_evals):
            raise DeftError('Input check failed. Z_evaluation_method must be in %s: Z_evaluation_method = %s' % (Z_evals,Z_eval))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure DT_MAX is valid
    try:
        if not isinstance(DT_MAX, utils.NUMBER):
            raise DeftError('Input check failed. Parameter "max_t_step" must be a number: max_t_step = %s' % type(DT_MAX))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (DT_MAX > 0):
            raise DeftError('Input check failed. Parameter "max_t_step" must be > 0: max_t_step = %s' % DT_MAX)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure print_t is valid
    try:
        if not isinstance(print_t, bool):
            raise DeftError('Input check failed. Parameter "print_t" must be a boolean: print_t = %s' % type(print_t))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure tolerance is valid
    try:
        if not isinstance(tollerance, float):
            raise DeftError('Input check failed. Parameter "tolerance" must be a float: tolerance = %s' % type(tollerance))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (tollerance > 0):
            raise DeftError('Input check failed. Parameter "tolerance" must be > 0: tolerance = %s' % tollerance)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure resolution is valid
    try:
        if not isinstance(resolution, utils.NUMBER):
            raise DeftError('Input check failed. Parameter "resolution" must be a number: resolution = %s' % type(resolution))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (resolution > 0):
            raise DeftError('Input check failed. Parameter "resolution" must be > 0: resolution = %s' % resolution)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure deft_seed is valid
    try:
        if deft_seed is not None and not isinstance(deft_seed, int):
            raise DeftError('Input check failed. Parameter "seed" must be either None or an integer: seed = %s' % type(deft_seed))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if (deft_seed is not None) and ((deft_seed < 0) or (deft_seed > 2**32-1)):
            raise DeftError('Input checks failed. Parameter "Seed" must be 0 <= deft_seed <= 2**32-1: deft_seed = %s' % deft_seed)
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure pt_method is valid
    pt_methods = [None, 'Lap', 'Lap+Imp']

    try:
        if not (pt_method in pt_methods):
            raise DeftError('Input check failed. posterior_sampling_method must be in %s: posterior_sampling_method = %s' % (pt_methods,pt_method))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure fix_t_at_t_star is valid

    try:
        if not isinstance(fix_t_at_t_star, bool):
            raise DeftError('Input check failed. Parameter "sample_only_at_l_star" must be a boolean: sample_only_at_l_star = %s' % type(fix_t_at_t_star))
    except DeftError as e:
        print(e)
        sys.exit(1)

    # Make sure num_pt_samples is valid
    try:
        if not isinstance(num_pt_samples, int):
            raise DeftError('Input check failed. Parameter "num_posterior_samples" must be an integer: num_posterior_samples = %s' % type(num_pt_samples))
    except DeftError as e:
        print(e)
        sys.exit(1)

    try:
        if not (num_pt_samples >= 0):
            raise DeftError('Input check failed. Parameter "num_posterior_samples" must be >= 0: num_posterior_samples = %s' % num_pt_samples)
    except DeftError as e:
        print(e)
        sys.exit(1)


    # ensure that max_log_evidence_ratio is a float
    try:
        if not isinstance(max_log_evidence_ratio_drop, utils.NUMBER):
            raise DeftError('Input check failed. Parameter "max_log_evidence_ratio" must be a number: tolerance = %s' % type(max_log_evidence_ratio_drop))
    except DeftError as e:
        print(e)
        sys.exit(1)
