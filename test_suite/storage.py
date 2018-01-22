"""
This is a place to store old versions of the codes in supplements.py
"""


# Laplace approach with importance sampling at maxent
def Laplace_at_maxent(phi_M, R, Delta, N, pt_sampling=False, num_samples=100, num_grid=400):
    G = len(phi_M)
    alpha = Delta._kernel_dim

    # Evaluate the transformation matrix
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
    M_mat = diags(sp.exp(-phi_M), 0).todense() * (N/G)
    M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
    U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
    U_mat = sp.mat(kernel_basis) * U_mat_on_kernel[1]

    lambdas = sp.exp(-phi_M) * (N/G)

    # For each eigen-component, draw y samples according to the Laplace distribution
    y_samples = np.zeros([alpha,num_samples])
    for i in range(alpha):
        eig_val = U_mat_on_kernel[0][i]
        eig_vec = sp.array(sp.mat(kernel_basis)*U_mat_on_kernel[1][:,i]).ravel()
        sigma = 1.0 / sp.sqrt(eig_val)
        # Determine lower and upper bounds
        N_lb = 0
        while distribution(eig_val, eig_vec, N_lb*sigma, lambdas, switch=0) > 1E-6:
            N_lb -= 1
        N_ub = 0
        while distribution(eig_val, eig_vec, N_ub*sigma, lambdas, switch=0) > 1E-6:
            N_ub += 1
        # Create a grid
        y_grid = sp.linspace(N_lb*sigma, N_ub*sigma, num_grid)
        # Evaluate probability at each bin center and draw samples
        h = y_grid[1] - y_grid[0]
        y_grid_bc = sp.linspace(N_lb*sigma+h/2, N_ub*sigma-h/2, num_grid-1)
        grid_bc_dist = np.zeros(num_grid-1)
        for j in range(num_grid-1):
            grid_bc_dist[j] = distribution(eig_val, eig_vec, y_grid_bc[j], lambdas, switch=0)
        prob = grid_bc_dist / sp.sum(grid_bc_dist)
        y_samples[i,:] = np.random.choice(y_grid_bc, num_samples, p=prob)
        # Randomly distribute the samples within each bin
        grid_dist = np.zeros(num_grid)
        for j in range(num_grid):
            grid_dist[j] = distribution(eig_val, eig_vec, y_grid[j], lambdas, switch=0)
        for k in range(num_samples):
            y_bc = y_samples[i,k]
            a = y_bc - h/2
            b = y_bc + h/2
            fa = grid_dist[int((a-N_lb*sigma)/h)]
            fb = grid_dist[int((b-N_lb*sigma)/h)]
            r = np.random.rand()
            if fa == fb:
                y_samples[i,k] = a + h * r
            else:
                h_ratio = (sp.sqrt(fa**2 + r*(fb**2 - fa**2)) - fa) / (fb - fa)
                y_samples[i,k] = a + h * h_ratio

    # Transform y samples to x samples
    x_samples = sp.array(U_mat*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        for k in range(num_samples):
            phi_samples[:,k] = x_samples[:,k] + phi_M
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S = Fxn(x_vec, lambdas)
            phi_weights[k] = sp.exp(-S)
        return phi_samples, phi_weights

    # For each sample, evaluate the object function
    q_samples = np.zeros(num_samples)
    for k in range(num_samples):
        y_vec = y_samples[:,k]
        x_vec = x_samples[:,k]
        S = Fxn(x_vec, lambdas)
        q_samples[k] = sp.exp(-S)
    eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the sample mean and effective sample size
    return sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)


# Laplace approach with importance sampling at t
def Laplace_at_t(phi_t, R, Delta, t, N, pt_sampling=False, num_samples=100, num_grid=400):
    G = len(phi_t)

    # Evaluate the transformation matrix
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    U_mat = np.linalg.eigh(A_mat)

    lambdas = sp.exp(-phi_t) * (N/G)

    # For each eigen-component, draw y samples according to the Laplace distribution
    y_samples = np.zeros([G,num_samples])
    for i in range(G):
        eig_val = U_mat[0][i]
        eig_vec = sp.array(U_mat[1][:,i]).ravel()
        sigma = 1.0 / sp.sqrt(eig_val)
        # Determine lower and upper bounds
        N_lb = 0
        while distribution(eig_val, eig_vec, N_lb*sigma, lambdas, switch=0) > 1E-6:
            N_lb -= 1
        N_ub = 0
        while distribution(eig_val, eig_vec, N_ub*sigma, lambdas, switch=0) > 1E-6:
            N_ub += 1
        # Create a grid
        y_grid = sp.linspace(N_lb*sigma, N_ub*sigma, num_grid)
        # Evaluate probability at each bin center and draw samples
        h = y_grid[1] - y_grid[0]
        y_grid_bc = sp.linspace(N_lb*sigma+h/2, N_ub*sigma-h/2, num_grid-1)
        grid_bc_dist = np.zeros(num_grid-1)
        for j in range(num_grid-1):
            grid_bc_dist[j] = distribution(eig_val, eig_vec, y_grid_bc[j], lambdas, switch=0)
        prob = grid_bc_dist / sp.sum(grid_bc_dist)
        y_samples[i,:] = np.random.choice(y_grid_bc, num_samples, p=prob)
        # Randomly distribute the samples within each bin
        grid_dist = np.zeros(num_grid)
        for j in range(num_grid):
            grid_dist[j] = distribution(eig_val, eig_vec, y_grid[j], lambdas, switch=0)
        for k in range(num_samples):
            y_bc = y_samples[i,k]
            a = y_bc - h/2
            b = y_bc + h/2
            fa = grid_dist[int((a-N_lb*sigma)/h)]
            fb = grid_dist[int((b-N_lb*sigma)/h)]
            r = np.random.rand()
            if fa == fb:
                y_samples[i,k] = a + h * r
            else:
                h_ratio = (sp.sqrt(fa**2 + r*(fb**2 - fa**2)) - fa) / (fb - fa)
                y_samples[i,k] = a + h * h_ratio

    # Transform y samples to x samples
    x_samples = sp.array(U_mat[1]*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        for k in range(num_samples):
            phi_samples[:,k] = x_samples[:,k] + phi_t
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S = Fxn(x_vec, lambdas)
            phi_weights[k] = sp.exp(-S)
        return phi_samples, phi_weights

    # For each sample, evaluate the object function
    q_samples = np.zeros(num_samples)
    for k in range(num_samples):
        y_vec = y_samples[:,k]
        x_vec = x_samples[:,k]
        S = Fxn(x_vec, lambdas)
        q_samples[k] = sp.exp(-S)
    eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the sample mean and effective sample size
    return sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)


# Generalized Laplace approach with importance sampling at maxent
def GLaplace_at_maxent(phi_M, R, Delta, N, sampling=True, pt_sampling=False, num_samples=100, num_grid=400):
    G = len(phi_M)
    alpha = Delta._kernel_dim

    # Evaluate the transformation matrix
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
    M_mat = diags(sp.exp(-phi_M), 0).todense() * (N/G)
    M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
    U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
    U_mat = sp.mat(kernel_basis) * U_mat_on_kernel[1]

    lambdas = sp.exp(-phi_M) * (N/G)

    # For each eigen-component, calculate the ratio of the areas and draw y samples according to
    # the generalized Laplace distribution
    gammas = np.zeros(alpha)
    y_samples = np.zeros([alpha,num_samples])
    for i in range(alpha):
        eig_val = U_mat_on_kernel[0][i]
        eig_vec = sp.array(sp.mat(kernel_basis)*U_mat_on_kernel[1][:,i]).ravel()
        sigma = 1.0 / sp.sqrt(eig_val)
        # Evaluate area under the Laplace distribution
        Lap_N_lb = 0
        while distribution(eig_val, eig_vec, Lap_N_lb*sigma, lambdas, switch=0) > 1E-6:
            Lap_N_lb -= 1
        Lap_N_ub = 0
        while distribution(eig_val, eig_vec, Lap_N_ub*sigma, lambdas, switch=0) > 1E-6:
            Lap_N_ub += 1
        Lap_y_grid = sp.linspace(Lap_N_lb*sigma, Lap_N_ub*sigma, num_grid)
        Lap_grid_dist = np.zeros(num_grid)
        for j in range(num_grid):
            Lap_grid_dist[j] = distribution(eig_val, eig_vec, Lap_y_grid[j], lambdas, switch=0)
        Lap_area = np.trapz(Lap_grid_dist, Lap_y_grid)
        # Evaluate area under the generalized Laplace distribution
        GLap_N_lb = 0
        while distribution(eig_val, eig_vec, GLap_N_lb*sigma, lambdas, switch=1) > 1E-6:
            GLap_N_lb -= 1
        GLap_N_ub = 0
        while distribution(eig_val, eig_vec, GLap_N_ub*sigma, lambdas, switch=1) > 1E-6:
            GLap_N_ub += 1
        GLap_y_grid = sp.linspace(GLap_N_lb*sigma, GLap_N_ub*sigma, num_grid)
        GLap_grid_dist = np.zeros(num_grid)
        for j in range(num_grid):
            GLap_grid_dist[j] = distribution(eig_val, eig_vec, GLap_y_grid[j], lambdas, switch=1)
        GLap_area = np.trapz(GLap_grid_dist, GLap_y_grid)
        # Take the ratio of the areas
        gammas[i] = GLap_area / Lap_area

        if sampling:
            N_lb = GLap_N_lb
            N_ub = GLap_N_ub
            y_grid = GLap_y_grid
            grid_dist = GLap_grid_dist
            # Evaluate probability at each bin center and draw samples
            h = y_grid[1] - y_grid[0]
            y_grid_bc = sp.linspace(N_lb*sigma+h/2, N_ub*sigma-h/2, num_grid-1)
            bc_dist = np.zeros(num_grid-1)
            for j in range(num_grid-1):
                bc_dist[j] = distribution(eig_val, eig_vec, y_grid_bc[j], lambdas, switch=1)
            prob = bc_dist / sp.sum(bc_dist)
            y_samples[i,:] = np.random.choice(y_grid_bc, num_samples, p=prob)
            # Randomly distribute the samples within each bin
            for k in range(num_samples):
                y_bc = y_samples[i,k]
                a = y_bc - h/2
                b = y_bc + h/2
                fa = grid_dist[int((a-N_lb*sigma)/h)]
                fb = grid_dist[int((b-N_lb*sigma)/h)]
                r = np.random.rand()
                if fa == fb:
                    y_samples[i,k] = a + h * r
                else:
                    h_ratio = (sp.sqrt(fa**2 + r*(fb**2 - fa**2)) - fa) / (fb - fa)
                    y_samples[i,k] = a + h * h_ratio

    # Transform y samples to x samples
    x_samples = sp.array(U_mat*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        for k in range(num_samples):
            phi_samples[:,k] = x_samples[:,k] + phi_M
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(alpha):
                eig_vec = sp.array(U_mat[:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            phi_weights[k] = sp.exp(-(S1-S2))
        return phi_samples, phi_weights

    if sampling:
        # For each sample, evaluate the object function
        q_samples = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(alpha):
                eig_vec = sp.array(U_mat[:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            q_samples[k] = sp.exp(-(S1-S2))
        eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the area ratios, the sample mean, and effective sample size
    if sampling:
        return sp.sum(sp.log(gammas))+sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)
    else:
        return sp.sum(sp.log(gammas))


# Generalized Laplace approach with importance sampling at t
def GLaplace_at_t(phi_t, R, Delta, t, N, sampling=True, pt_sampling=False, num_samples=100, num_grid=400):
    G = len(phi_t)

    # Evaluate the transformation matrix
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    U_mat = np.linalg.eigh(A_mat)

    lambdas = sp.exp(-phi_t) * (N/G)

    # For each eigen-component, calculate the ratio of the areas and draw y samples according to
    # the generalized Laplace distribution
    gammas = np.zeros(G)
    y_samples = np.zeros([G,num_samples])
    for i in range(G):
        eig_val = U_mat[0][i]
        eig_vec = sp.array(U_mat[1][:,i]).ravel()
        sigma = 1.0 / sp.sqrt(eig_val)
        # Find the lower and upper bounds of the Laplace distribution and tabulate its values
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
        # See if these two distributions are close enough
        if Lap_Ns == GLap_Ns:
            diff_Es = abs(sp.array(Lap_Es) - sp.array(GLap_Es))
            if all(diff_Es < 1E-6):
                do_integration = False
            else:
                do_integration = True
        else:
            do_integration = True
        # If these two distributions are close enough, set gamma to 1, otherwise do the integration
        if not do_integration:
            gammas[i] = 1.0
        else:
            # Evaluate area under the Laplace distribution
            Lap_y_grid = sp.linspace(Lap_Ns[0]*sigma, Lap_Ns[-1]*sigma, num_grid)
            Lap_grid_dist = np.zeros(num_grid)
            for j in range(num_grid):
                Lap_grid_dist[j] = distribution(eig_val, eig_vec, Lap_y_grid[j], lambdas, switch=0)
            Lap_area = np.trapz(Lap_grid_dist, Lap_y_grid)
            # Evaluate area under the generalized Laplace distribution
            GLap_y_grid = sp.linspace(GLap_Ns[0]*sigma, GLap_Ns[-1]*sigma, num_grid)
            GLap_grid_dist = np.zeros(num_grid)
            for j in range(num_grid):
                GLap_grid_dist[j] = distribution(eig_val, eig_vec, GLap_y_grid[j], lambdas, switch=1)
            GLap_area = np.trapz(GLap_grid_dist, GLap_y_grid)
            # Take the ratio of the areas
            gammas[i] = GLap_area / Lap_area

        if sampling:
            if not do_integration:
                switch = 0
                N_lb = Lap_N_lb
                N_ub = Lap_N_ub
            else:
                switch = 1
                N_lb = GLap_N_lb
                N_ub = GLap_N_ub
            # Create a grid
            y_grid = sp.linspace(N_lb*sigma, N_ub*sigma, num_grid)
            # Evaluate probability at each bin center and draw samples
            h = y_grid[1] - y_grid[0]
            y_grid_bc = sp.linspace(N_lb*sigma+h/2, N_ub*sigma-h/2, num_grid-1)
            grid_bc_dist = np.zeros(num_grid-1)
            for j in range(num_grid-1):
                grid_bc_dist[j] = distribution(eig_val, eig_vec, y_grid_bc[j], lambdas, switch=switch)
            prob = grid_bc_dist / sp.sum(grid_bc_dist)
            y_samples[i,:] = np.random.choice(y_grid_bc, num_samples, p=prob)
            # Randomly distribute the samples within each bin
            grid_dist = np.zeros(num_grid)
            for j in range(num_grid):
                grid_dist[j] = distribution(eig_val, eig_vec, y_grid[j], lambdas, switch=switch)
            for k in range(num_samples):
                y_bc = y_samples[i,k]
                a = y_bc - h/2
                b = y_bc + h/2
                fa = grid_dist[int((a-N_lb*sigma)/h)]
                fb = grid_dist[int((b-N_lb*sigma)/h)]
                r = np.random.rand()
                if fa == fb:
                    y_samples[i,k] = a + h * r
                else:
                    h_ratio = (sp.sqrt(fa**2 + r*(fb**2 - fa**2)) - fa) / (fb - fa)
                    y_samples[i,k] = a + h * h_ratio

    # Transform y samples to x samples
    x_samples = sp.array(U_mat[1]*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        for k in range(num_samples):
            phi_samples[:,k] = x_samples[:,k] + phi_t
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(G):
                eig_vec = sp.array(U_mat[1][:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            phi_weights[k] = sp.exp(-(S1-S2))
        return phi_samples, phi_weights

    if sampling:
        # For each sample, evaluate the object function
        q_samples = np.zeros(num_samples)
        for k in range(num_samples):
            y_vec = y_samples[:,k]
            x_vec = x_samples[:,k]
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(G):
                eig_vec = sp.array(U_mat[1][:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            q_samples[k] = sp.exp(-(S1-S2))
        eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the area ratios, the sample mean, and effective sample size
    if sampling:
        return sp.sum(sp.log(gammas))+sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)
    else:
        return sp.sum(sp.log(gammas))


# Laplace approach with importance sampling at t
def para_Laplace_at_t(phi_t, R, Delta, t, N, go_parallel, pt_sampling=False, num_samples=100, num_grid=400):

    # Evaluate the transformation matrix
    G = len(phi_t)
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    U_mat = np.linalg.eigh(A_mat)
    lambdas = sp.exp(-phi_t) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, draw y samples according to the corresponding Laplace distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), itertools.repeat(num_grid),
                                U_mat[0], sp.array(U_mat[1].T), itertools.repeat(lambdas))
        outputs = pool.map(y_sampling_of_Lap, inputs)
        y_samples = sp.array(outputs)
    else:
        y_samples = np.zeros([G,num_samples])
        for i in range(G):
            inputs = [num_samples, num_grid, U_mat[0][i], sp.array(U_mat[1][:,i].T), lambdas]
            outputs = y_sampling_of_Lap(inputs)
            y_samples[i,:] = outputs

    # Transform y samples to x samples
    x_samples = sp.array(U_mat[1]*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            x_vec = x_samples[:,k]
            phi_samples[:,k] = x_vec + phi_t
            S = Fxn(x_vec, lambdas)
            phi_weights[k] = sp.exp(-S)
        return phi_samples, phi_weights

    # For each sample, evaluate the object function
    if go_parallel:
        inputs = itertools.izip(sp.array(sp.mat(x_samples).T), itertools.repeat(lambdas))
        outputs = pool.map(obj_fxn_evaluation_of_Lap, inputs)
        q_samples = sp.array(outputs)
    else:
        q_samples = np.zeros(num_samples)
        for k in range(num_samples):
            inputs = [x_samples[:,k], lambdas]
            outputs = obj_fxn_evaluation_of_Lap(inputs)
            q_samples[k] = outputs

    # Calculate effective sample size
    eff_sample_size = int(sp.sum(q_samples)**2/sp.sum(q_samples**2))

    # Return the sample mean and effective sample size
    return sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)


# Laplace approach with importance sampling at maxent
def para_Laplace_at_maxent(phi_M, R, Delta, N, go_parallel, pt_sampling=False, num_samples=100, num_grid=400):

    # Evaluate the transformation matrix
    G = len(phi_M)
    alpha = Delta._kernel_dim
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
    M_mat = diags(sp.exp(-phi_M), 0).todense() * (N/G)
    M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
    U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
    U_mat = sp.mat(kernel_basis) * U_mat_on_kernel[1]
    lambdas = sp.exp(-phi_M) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, draw y samples according to the corresponding Laplace distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), itertools.repeat(num_grid),
                                U_mat_on_kernel[0], sp.array(U_mat.T), itertools.repeat(lambdas))
        outputs = pool.map(y_sampling_of_Lap, inputs)
        y_samples = sp.array(outputs)
    else:
        y_samples = np.zeros([alpha,num_samples])
        for i in range(alpha):
            inputs = [num_samples, num_grid, U_mat_on_kernel[0][i], sp.array(U_mat[:,i].T), lambdas]
            outputs = y_sampling_of_Lap(inputs)
            y_samples[i,:] = outputs

    # Transform y samples to x samples
    x_samples = sp.array(U_mat*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            x_vec = x_samples[:,k]
            phi_samples[:,k] = x_vec + phi_M
            S = Fxn(x_vec, lambdas)
            phi_weights[k] = sp.exp(-S)
        return phi_samples, phi_weights

    # For each sample, evaluate the object function
    if go_parallel:
        inputs = itertools.izip(sp.array(sp.mat(x_samples).T), itertools.repeat(lambdas))
        outputs = pool.map(obj_fxn_evaluation_of_Lap, inputs)
        q_samples = sp.array(outputs)
    else:
        q_samples = np.zeros(num_samples)
        for k in range(num_samples):
            inputs = [x_samples[:,k], lambdas]
            outputs = obj_fxn_evaluation_of_Lap(inputs)
            q_samples[k] = outputs

    # Calculate effective sample size
    eff_sample_size = int(sp.sum(q_samples)**2/sp.sum(q_samples**2))

    # Return the sample mean and effective sample size
    return sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)


# Generalized Laplace approach with importance sampling at t
def para_GLaplace_at_t(phi_t, R, Delta, t, N, go_parallel, sampling=True, pt_sampling=False, num_samples=100, num_grid=400):

    # Evaluate the transformation matrix
    G = len(phi_t)
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    U_mat = np.linalg.eigh(A_mat)
    lambdas = sp.exp(-phi_t) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, calculate the area ratio and draw y samples according to
    # the corresponding generalized Laplace distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), itertools.repeat(num_grid), U_mat[0],
                                sp.array(U_mat[1].T), itertools.repeat(lambdas), itertools.repeat(sampling))
        outputs = pool.map(y_sampling_of_GLap, inputs)
        gammas = np.zeros(G)
        y_samples = np.zeros([G,num_samples])
        for i in range(G):
            gammas[i] = outputs[i][0]
            if sampling:
                y_samples[i,:] = outputs[i][1]
    else:
        gammas = np.zeros(G)
        y_samples = np.zeros([G,num_samples])
        for i in range(G):
            inputs = [num_samples, num_grid, U_mat[0][i], sp.array(U_mat[1][:,i].T), lambdas, sampling]
            outputs = y_sampling_of_GLap(inputs)
            gammas[i] = outputs[0]
            if sampling:
                y_samples[i,:] = outputs[1]

    # Transform y samples to x samples
    x_samples = sp.array(U_mat[1]*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            x_vec = x_samples[:,k]
            y_vec = y_samples[:,k]
            phi_samples[:,k] = x_vec + phi_t
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(G):
                eig_vec = sp.array(U_mat[1][:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            phi_weights[k] = sp.exp(-(S1-S2))
        return phi_samples, phi_weights

    if sampling:
        # For each sample, evaluate the object function
        if go_parallel:
            inputs = itertools.izip(sp.array(sp.mat(x_samples).T), sp.array(sp.mat(y_samples).T),
                                    itertools.repeat(lambdas), itertools.repeat(sp.array(U_mat[1])))
            outputs = pool.map(obj_fxn_evaluation_of_GLap, inputs)
            q_samples = sp.array(outputs)
        else:
            q_samples = np.zeros(num_samples)
            for k in range(num_samples):
                inputs = [x_samples[:,k], y_samples[:,k], lambdas, sp.array(U_mat[1])]
                outputs = obj_fxn_evaluation_of_GLap(inputs)
                q_samples[k] = outputs
        # Calculate effective sample size
        eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the area ratios, the sample mean, and effective sample size
    if sampling:
        return sp.sum(sp.log(gammas))+sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)
    else:
        return sp.sum(sp.log(gammas)), 'N/A', 1.0


# Generalized Laplace approach with importance sampling at maxent
def para_GLaplace_at_maxent(phi_M, R, Delta, N, go_parallel, sampling=True, pt_sampling=False, num_samples=100, num_grid=400):

    # Evaluate the transformation matrix
    G = len(phi_M)
    alpha = Delta._kernel_dim
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
    M_mat = diags(sp.exp(-phi_M), 0).todense() * (N/G)
    M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
    U_mat_on_kernel = np.linalg.eigh(M_mat_on_kernel)
    U_mat = sp.mat(kernel_basis) * U_mat_on_kernel[1]
    lambdas = sp.exp(-phi_M) * (N/G)

    # If requested to go parallel, set up a pool of workers for parallel computation
    if go_parallel:
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)

    # For each eigen-component, calculate the area ratio and draw y samples according to
    # the corresponding generalized Laplace distribution
    if go_parallel:
        inputs = itertools.izip(itertools.repeat(num_samples), itertools.repeat(num_grid), U_mat_on_kernel[0],
                                sp.array(U_mat.T), itertools.repeat(lambdas), itertools.repeat(sampling))
        outputs = pool.map(y_sampling_of_GLap, inputs)
        gammas = np.zeros(alpha)
        y_samples = np.zeros([alpha,num_samples])
        for i in range(alpha):
            gammas[i] = outputs[i][0]
            if sampling:
                y_samples[i,:] = outputs[i][1]
    else:
        gammas = np.zeros(alpha)
        y_samples = np.zeros([alpha,num_samples])
        for i in range(alpha):
            inputs = [num_samples, num_grid, U_mat_on_kernel[0][i], sp.array(U_mat[:,i].T), lambdas, sampling]
            outputs = y_sampling_of_GLap(inputs)
            gammas[i] = outputs[0]
            if sampling:
                y_samples[i,:] = outputs[1]

    # Transform y samples to x samples
    x_samples = sp.array(U_mat*sp.mat(y_samples))

    # If called from posterior sampling, return phi samples along with their weights
    if pt_sampling:
        phi_samples = np.zeros([G,num_samples])
        phi_weights = np.zeros(num_samples)
        for k in range(num_samples):
            x_vec = x_samples[:,k]
            y_vec = y_samples[:,k]
            phi_samples[:,k] = x_vec + phi_M
            S1 = Fxn(x_vec, lambdas)
            S2 = 0.0
            for i in range(alpha):
                eig_vec = sp.array(U_mat[:,i]).ravel()
                S2 += Fxn(eig_vec*y_vec[i], lambdas)
            phi_weights[k] = sp.exp(-(S1-S2))
        return phi_samples, phi_weights

    if sampling:
        # For each sample, evaluate the object function
        if go_parallel:
            inputs = itertools.izip(sp.array(sp.mat(x_samples).T), sp.array(sp.mat(y_samples).T),
                                    itertools.repeat(lambdas), itertools.repeat(sp.array(U_mat)))
            outputs = pool.map(obj_fxn_evaluation_of_GLap, inputs)
            q_samples = sp.array(outputs)
        else:
            q_samples = np.zeros(num_samples)
            for k in range(num_samples):
                inputs = [x_samples[:,k], y_samples[:,k], lambdas, sp.array(U_mat)]
                outputs = obj_fxn_evaluation_of_GLap(inputs)
                q_samples[k] = outputs
        # Calculate effective sample size
        eff_sample_size = int(sp.sum(q_samples)**2 / sp.sum(q_samples**2))

    # Return the area ratios, the sample mean, and effective sample size
    if sampling:
        return sp.sum(sp.log(gammas))+sp.log(sp.mean(q_samples)), eff_sample_size, sp.mean(q_samples)
    else:
        return sp.sum(sp.log(gammas)), 'N/A', 1.0


# Feynman diagram calculations at t
def Feynman_at_t(phi_t, R, Delta, t, N):
    G = len(phi_t)

    # Evaluate propagator matrix
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    P_mat = np.linalg.inv(A_mat)

    # Evaluate vertex vector
    V = sp.exp(-phi_t) * (N/G)

    # Calculate Feynman diagrams
    correction = diagrams_1st_order(G, P_mat, V)

    # Return
    return correction, 'N/A', 1.0


# Feynman diagram calculations at maxent
def Feynman_at_maxent(phi_M, Delta, N):
    G = len(phi_M)
    alpha = Delta._kernel_dim

    # Evaluate propagator matrix
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()
    M_mat = diags(sp.exp(-phi_M),0).todense() * (N/G)
    M_mat_on_kernel = sp.mat(kernel_basis).T * M_mat * sp.mat(kernel_basis)
    M_inv_on_kernel = sp.linalg.inv(M_mat_on_kernel)
    P_mat = sp.mat(kernel_basis) * M_inv_on_kernel * sp.mat(kernel_basis).T

    # Evaluate vertex vector
    V = sp.exp(-phi_M) * (N/G)

    # Calculate Feynman diagrams
    correction = diagrams_1st_order(G, P_mat, V)

    # Return
    return correction, 'N/A', 1.0


# Metropolis Monte Carlo at t
def Metropolis_at_t(phi_t, R, Delta, t, N, go_parallel, pt_sampling, num_samples):
    G = len(phi_t)
    num_thermalization_steps = 10*G
    num_steps_per_sample = G

    phi_samples = np.zeros([G,num_samples])
    sample_index = 0

    # Find eigen-modes of the Hessian matrix
    H = deft_core.hessian(phi_t, R, Delta, t, N)
    A_mat = H.todense() * (N/G)
    U_mat = np.linalg.eigh(A_mat)

    # Initialize
    phi_current = phi_t
    S_current = deft_core.action(phi_current, R, Delta, t, N) * (N/G)

    # Do Monte Carlo sampling
    for k in range(num_thermalization_steps + num_samples*num_steps_per_sample + 1):
        g = np.random.randint(0,G)
        eig_val = U_mat[0][g]
        eig_vec = sp.array(U_mat[1][:,g]).ravel()
        step_size = np.random.normal(0,1.0/np.sqrt(eig_val))
        step_vec = eig_vec * step_size
        phi_new = phi_current + step_vec
        S_new = deft_core.action(phi_new, R, Delta, t, N) * (N/G)
        if np.log(np.random.uniform(0,1)) < (S_current - S_new):
            phi_current = phi_new
            S_current = S_new
        if (k > num_thermalization_steps) and (k%num_steps_per_sample == 0):
            phi_samples[:,sample_index] = phi_current
            sample_index += 1

    # Return phi samples and (dummy) phi weights
    phi_weights = np.ones(num_samples)
    return phi_samples, phi_weights


# Metropolis Monte Carlo at maxent
def Metropolis_at_maxent(phi_M, R, Delta, N, go_parallel, pt_sampling, num_samples):
    G = len(phi_M)
    alpha = Delta._kernel_dim
    num_thermalization_steps = 10*G
    num_steps_per_sample = G

    phi_samples = np.zeros([G,num_samples])
    sample_index = 0

    # Find the kernel basis
    Delta_sparse = Delta.get_sparse_matrix()
    Delta_mat = Delta_sparse.todense() * (N/G)
    Delta_diagonalized = np.linalg.eigh(Delta_mat)
    kernel_basis = np.zeros([G,alpha])
    for i in range(alpha):
        kernel_basis[:,i] = Delta_diagonalized[1][:,i].ravel()

    # Find coefficients of phi_M in the kernel basis
    coeffs = np.zeros(alpha)
    for i in range(alpha):
        coeffs[i] = sp.mat(kernel_basis[:,i]) * sp.mat(phi_M).T

    # Find eigen-modes of the Hessian matrix
    H = maxent.hessian_per_datum_from_coeffs(coeffs, R, kernel_basis)
    A_mat = sp.mat(H) * N
    U_mat = np.linalg.eigh(A_mat)

    # Initialize
    coeffs_current = coeffs
    S_current = maxent.action_per_datum_from_coeffs(coeffs_current, R, kernel_basis) * N

    # Do Monte Carlo sampling
    for k in range(num_thermalization_steps + num_samples*num_steps_per_sample + 1):
        a = np.random.randint(0,alpha)
        eig_val = U_mat[0][a]
        eig_vec = sp.array(U_mat[1][:,a]).ravel()
        step_size = np.random.normal(0,1.0/np.sqrt(eig_val))
        step_vec = eig_vec * step_size
        coeffs_new = coeffs_current + step_vec
        S_new = maxent.action_per_datum_from_coeffs(coeffs_new, R, kernel_basis) * N
        if np.log(np.random.uniform(0,1)) < (S_current - S_new):
            coeffs_current = coeffs_new
            S_current = S_new
        if (k > num_thermalization_steps) and (k%num_steps_per_sample == 0):
            phi_samples[:,sample_index] = maxent.coeffs_to_field(coeffs_current, kernel_basis)
            sample_index += 1

    # Return phi samples and (dummy) phi weights
    phi_weights = np.ones(num_samples)
    return phi_samples, phi_weights
