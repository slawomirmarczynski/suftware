# Generally, Python programs should be written with the assumption that all
# users are consenting adults, and thus are responsible for using things
# correctly themselves. (Silas Ray, StackOverflow)

import numpy as np
import scipy as sp

# from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import det, eigh, qr
from scipy.special import comb

# ------- WTF?
#
# AT: fixing SciPy comb import bug: 8_19_2019
# for SciPy versions >= 0.19
# try:
#     from scipy.special import comb
# except ImportError:
#     from scipy.misc import comb



import utils


class Laplacian:
    """
    Class container for Laplacian operators. Constructor computes spectrum.

    @note: Something is inconsistent - discrete Laplace matrix is an graph
           theory concept. Here is "another Laplace matrix" - probably finite
           difference representation of nabla square differential operator?
    """

# operator_type --> dimension=1,2 ; periodic=True,False
# ..._gridpoints --> npts
# step = 1.0 wywalone zupełnie

    def __init__(self, dimension, periodic, alpha, npts, step=1.0):
        """


        Args:
            dimension (int): the Laplacian dimension, should be either 1 or 2.
            periodic (boolean): should be either True or False (for bilateral).
            alpha (int): the order of the operator.
            npts (int): the number of meshpoints across a given dimension.
            step (float): ???

        Raises:
            RuntimeError: DESCRIPTION.

        Returns:
            None.

        """

        # Reflections. Read-only in pythonic way: uppercase identifiers should
        # be always treated as constants by an gentelmen agreement.
        #
        self.DIMENSION = dimension
        self.PERIODIC = periodic
        self.ALPHA = alpha
        self.NPTS = npts
        self.STEP = step

        # Notice, that Python convention (use lowercase letters for variables)
        # and the common mathematical convention (use uppercase letters for
        # matrices) are contradictory. We stuck to Python PEP8.

        if dimension == 1:

            if periodic:

                # m is a matrix, mostly tridiagonal
                #
                m = (2 * np.eye(npts) 
                     - np.eye(npts, npts, -1)
                     - np.eye(npts, npts, +1))
                m[npts - 1, 0] = -1.0
                m[0, npts - 1] = -1.0
                step = (m / (step**2)) ** alpha

            else:

                rhs = np.eye(npts)

                # Multiply alpha derivative matrices of together. 
                # Reduce dimension going left
                #
                for a in range(alpha):
                    # Compute the derivative matrix (matrix of derivatives?)
                    #
                    m = np.eye(npts - a) - np.eye(npts - a, npts - a, -1)
                    rhs = (m[1:,:] / step) * rhs 

                # Construct final bilateral laplacian???
                step = rhs.T * rhs

            # x is a vector filled with values from -1 to +1 (inclusive).
            #
            x = np.linspace(-1.0, +1.0, npts)
            basis = np.zeros((npts, alpha))
            for i in range(alpha):
                m = np.eye(alpha, alpha, i - 1)
                basis[:,i] = np.polynomial.legendre.legval(x, m)
            basis = utils.normalize(basis, step)

            self.G = self._kernel_basis.shape[0]
            self._kernel_dim = self._kernel_basis.shape[1]

            return step, kernel_basis


        elif dimension == 2:
            self._sparse_matrix, self._kernel_basis = \
                laplacian_2d( num_gridpoints,
                              operator_order,
                              step,
                              periodic=periodic,
                              sparse=True,
                              report_kernel=True)

            self._Gx = int(num_gridpoints[0])
            self._Gy = int(num_gridpoints[1])
            self._G = self._Gx * self._Gy
            self._alpha = operator_order
            assert( self._G == self._kernel_basis.shape[0] )
            self._kernel_dim = self._kernel_basis.shape[1]

        else:
            raise ControlledError('/Laplacian/ Cannot identify operator_type: operator_type = %s' % operator_type)

        # Compute spectrum, and set lowest rank eigenvectors as kernel
        self._dense_matrix = self._sparse_matrix.todense()
        eigenvalues, eigenvectors = eigh(self._dense_matrix)
        self._eigenvalues = eigenvalues
        self._eigenbasis = utils.normalize(eigenvectors)
        #self._kernel_basis = self._eigenbasis[:,:self._kernel_dim]

        # Set kernel eigenvalues and eigenvectors
        self._eigenvalues[:self._kernel_dim] = 0.0
        self._eigenbasis[:,:self._kernel_dim] = self._kernel_basis

    def get_kernel_basis(self):
        """ Returns the kernel as a kernel_dim x npts numpy array """
        return sp.copy(self._kernel_basis)

    def get_kernel_dim(self):
        """ Return the dimension of the kernel of this operator. """
        return self._kernel_dim

    def get_sparse_matrix(self):
        """ Return a sparse matrix version of this operator. """
        return self._sparse_matrix

    def get_sparse_Lambda(self):
        """ Return a sparse matrix version of Lambda. """
        return self._sparse_matrix

    def get_dense_matrix(self):
        """ Return a dense matrix version of this operator. """
        return self._sparse_matrix.todense()

    def get_dense_Lambda(self):
        """ Return a dense matrix version of Lambda. """
        return self._sparse_matrix.todense()


def derivative_matrix_1d(npts, step):
.......



def laplacian_1d(npts, alpha, step, periodic, sparse=True, report_kernel=True):
    """ Returns a npts x npts sized 1d laplacian matrix of order alpha """

    # Make sure sparse is valid
    if not isinstance(sparse, bool):
        raise ControlledError('/laplacian_1d/ sparse must be a boolean: sparse = %s' % type(sparse))
    # Make sure report_kernel is valid
    if not isinstance(report_kernel, bool):
        raise ControlledError('/laplacian_1d/ report_kernel must be a boolean: report_kernel = %s' % type(report_kernel))

    x = (sp.arange(npts) - (npts-1)/2.)/(npts/2.)

    # If periodic boundary conditions, construct regular laplacian
    if periodic:
        tmp_mat = 2*sp.diag(sp.ones(npts),0) - sp.diag(sp.ones(npts-1),-1) - sp.diag(sp.ones(npts-1),+1)
        tmp_mat[npts-1,0] = -1.0
        tmp_mat[0,npts-1] = -1.0
        step = (sp.mat(tmp_mat)/(step**2))**alpha

        # Get kernel, which is just the constant vector v = sp.ones([npts,1])
        # kernel_basis = utils.normalize(v, step)
        kernel_basis = utils.legendre_basis_1d(npts, 1, step)

    # Otherwise, construct bilateral laplacian
    else:

        # Initialize to npts x npts identity matrix
        right_side = sp.diag(sp.ones(npts),0)

        # Multiply alpha derivative matrices of together. Reduce dimension going left
        for a in range(alpha):
            right_side = derivative_matrix_1d(npts-a, step)*right_side

        # Construct final bilateral laplacian
        step = right_side.T*right_side

        # Make sure step is valid
        if not (step.shape[0] == step.shape[1] == npts):
            raise ControlledError('/laplacian_1d/ step must have shape (%d, %d): step.shape = %s' % (npts, npts, step.shape))

        # Construct a basis for the kernel from legendre polynomials
        kernel_basis = utils.legendre_basis_1d(npts, alpha, step)

        # Make sure kernel_basis is valid
        if not ((kernel_basis.shape[0] == npts) and (kernel_basis.shape[1] == alpha)):
            raise ControlledError('/laplacian_1d/ kernel_basis must have shape (%d, %d): kernel_basis.shape = %s' %
                                  (npts,alpha,kernel_basis.shape))

    # Sparsify matrix if requested
    if sparse:
        step = csr_matrix(step)

    # Report kernel if requested
    if report_kernel:
        return step, kernel_basis

    # Otherwise, just report matrix
    else:
        return step


def laplacian_2d( num_gridpoints, alpha, step=[1.0,1.0], periodic=False, sparse=False, report_kernel=False):
    """ Returns a GxG (npts=GxGy) sized 2d Laplacian """
    assert(len(num_gridpoints)==2)
    Gx = num_gridpoints[0]
    Gy = num_gridpoints[1]
    npts = Gx*Gy
    assert(Gx == int(Gx))
    assert(Gy == int(Gy))
    assert(alpha == int(alpha))
    assert(alpha >= 1)
    assert(len(step)==2)
    assert(type(step[0]) == float)
    assert(type(step[1]) == float)
    hx = step[0]
    hy = step[0]
    assert(hx > 0.0)
    assert(hy > 0.0)

    # Identity matrices, which will be used below
    I_x = sp.mat(sp.identity(Gx))
    I_y = sp.mat(sp.identity(Gy))

    # Compute x-coords and y-coords
    x = (sp.arange(Gx) - (Gx-1)/2.)/(Gx/2.)
    y_grid = (sp.arange(Gy) - (Gy-1)/2.)/(Gy/2.)
    xs,ys = np.meshgrid(x,y_grid)

    # If periodic boundary conditions,
    if periodic:
        step_x = laplacian_1d(Gx, alpha=1, step=hx, periodic=True)
        step_y = laplacian_1d(Gy, alpha=1, step=hy, periodic=True)

        # Use the kroneker product to generate a first-order operator
        step_1 = sp.mat(sp.kron(step_x,I_y) + sp.kron(I_x,step_y))

        # Raise operator to alpha power
        step = step_1**alpha

    # If bilateral, construct alpha-order bilateral laplacian algorithmically
    else:
        step_x_array = [I_x]
        step_y_array = [I_y]

        for a in range(1,alpha+1):
            step_x_array.append( laplacian_1d(Gx, alpha=a, step=hx) )
            step_y_array.append( laplacian_1d(Gy, alpha=a, step=hy) )

        for a in range(alpha+1):
            Dx = step_x_array[alpha-a]
            Dy = step_y_array[a]
            coeff = comb(alpha,a)
            if a == 0:
                step = coeff*sp.mat(sp.kron(Dx,Dy))
            else:
                step += coeff*sp.mat(sp.kron(Dx,Dy))

    # Build kernel from 2d legendre polynomials
    if periodic:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, 1, step)
    else:
        kernel_basis = utils.legendre_basis_2d(Gx, Gy, alpha, step)

    # Sparsify matrix if requested
    if sparse:
        step = sp.sparse.csr_matrix(step)

    # Report kernel if requested
    if report_kernel:
        return step, kernel_basis

    # Otherwise, just report matrix
    else:
        return step
