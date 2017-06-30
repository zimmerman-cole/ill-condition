import numpy as np
import numpy.linalg as la
import traceback, sys, scipy, time, sklearn
from scipy import optimize as scopt
from collections import OrderedDict
from sklearn.linear_model import SGDClassifier


# TODO: currently most (?) methods calculate residual an extra time each iteration
#               (when checking if resid norm is within tolerance range)

# || b - Ax ||
def norm_dif(x, *args):
    """
    Return || b - Ax || (Frobenius norm).
    """
    A, b = args
    return la.norm(b - np.dot(A, x))

class GradientDescent:
    """
    Gradient descent solver.
    """

    def __init__(self, A=None, b=None, full_output=False):
        self.A = A
        self.b = b
        self.full_output = full_output

    def __str__(self):
        l1 = 'Gradient Descent Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def __repr__(self):
        return self.__str__()

    def fit(self, A, b):
        self.A = A
        self.b = b

    def gradient_descent(self, tol=10**-5, x0 = None, max_iter = 500, recalc=20, x_true=None):
        """
        For SYMMETRIC, POSITIVE-DEFINITE matrices.

        If self.full_output and x_true != None, this returns:
            xopt, n_iter, resids, path, x_difs
        If self.full_output and x_true is None:
            xopt, n_iter, resids, path
        Else:
            xopt

        TODO: make separate method for returning path?
        TODO: is slightly less accurate than gd_alt and gd_nonsymm
                (but much faster than them)
        """
        if self.A is None or self.b is None:
            raise AttributeError('A and/or b haven\'t been set yet.')

        assert len(self.A) == len(self.A.T) == len(self.b)
        if x0 is None:
            x = np.zeros(len(self.A))
        else:
            x = np.copy(x0)

        if self.full_output:
            return self.__gd_full(tol=tol, x=x, max_iter=max_iter, recalc=recalc, x_true=x_true)
        else:
            return self.__gd_bare(tol=tol, x=x, max_iter=max_iter, recalc=recalc)

    def __gd_full(self, tol, x, max_iter, recalc, x_true):
        """
        Tracks everything (times/iteration, residuals, path taken, etc.).

        If you provide an x_true, it also tracks ||x - x_true|| at each
            iteration.
        """
        start_time = time.time()
        path = [x]      # track path taken
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]

        # First descent step ======================================
        r = self.b - np.dot(self.A, x)
        r_norm = la.norm(r)
        residuals = [(r_norm, time.time() - start_time)]

        # Check if close enough already
        if r_norm <= tol:
            if x_true is None:
                return x, i, residuals, path
            else:
                return x, i, residuals, path, x_difs

        # If not, take a step
        i = 1
        Ar = np.dot(self.A, r)
        a = np.inner(r.T, r) / np.dot(r.T, Ar)
        x += a * r
        path.append(x)
        if x_true is not None:
            x_difs.append(la.norm(x - x_true))
        # =========================================================

        # Rest of descent
        while i < max_iter:
            # Directly calculate residual every 'recalc' steps
            if (i % recalc) == 0:
                r = self.b - np.dot(self.A, x)
            else:
                # Else, update using one less matrix-vector product
                r -= a * Ar
            r_norm = la.norm(r)
            residuals.append((r_norm, time.time() - start_time))

            # Check if close enough
            if r_norm <= tol: break
            i += 1

            # If not, take another step
            Ar = np.dot(self.A, r)
            x += a * r
            path.append(x)
            if x_true is not None:
                x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals, path
        else:
            return x, i, residuals, path, x_difs

    def __gd_bare(self, tol, x, max_iter, recalc):
        """
        For max performance.
        """
        # First descent step ==========================================
        r = self.b - np.dot(self.A, x)
        # Check if close enough already
        if la.norm(r) <= tol:
            return x

        # If not, take a step
        Ar = np.dot(self.A, r)
        a = np.inner(r.T, r) / np.dot(r.T, Ar)
        x += a * r
        # ==============================================================

        for i in range(1, max_iter):
            # Directly calculate residual every 'recalc' steps
            if (i % recalc) == 0:
                r = self.b - np.dot(self.A, x)
            else:
                # Else, update using one less matrix-vector product
                r -= a * Ar

            # Check if close enough
            if r_norm <= tol: break

            # If not, take another step
            Ar = np.dot(self.A, r)
            x += a * r

        return x

# for symmetric, positive-definite A
def conjugate_gradient_ideal(A, b, tol=0.001, x = None, numIter = 500, full_output=False):
    """
    For SYMMETRIC, POSITIVE-DEFINITE matrices.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (p. 32)

    Tested on a handful of small (~50x50 - 500x500 matrices) w/ various
    condition numbers. Behaviour is as expected - systems with higher
    condition numbers take longer to solve accurately.

    TODO: fix residual error accumulation

    Returns:
        If not full_output: just the optimal x.
        If full_output: optimal x, num iterations taken, success, residuals plot.
    """
    #tol *= la.norm(A)

    m, n = len(A), len(A.T)

    if x is None:
        x = np.zeros(n)

    # d: first search direction (same as initial residual)
    d = b - np.dot(A, x) # d(0) = r(0) = b - Ax(0)
    r = d                # from eq. (45)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        if full_output:
            resids[time.time() - start_time] = norm_dif(x, A, b)

        # TODO: recalculate residual here every _ iters to avoid accumulating error
        # if 0:
        #     print(('r(%d): ' + str(r)) % i)
        #     recalc_r = b - np.dot(A, x)
        #     print('recalc: ' + str(recalc_r))
        #     print('resid dif: %f' % la.norm(r - recalc_r))


        a = np.dot(r.T, r) / np.dot(d.T, np.dot(A, d)) # eq. (46)

        x += a * d

        new_r = r - (a * np.dot(A, d)) # calculate new residual (A-orthogonal to
                                       # previous except d)      (eq. 47)

        beta = np.dot(new_r.T, new_r) / np.dot(r.T, r) # eq. (48)

        d = new_r + beta * d
        r = new_r

        if la.norm(b - np.dot(A, x)) < tol:
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

def conjugate_gradient_psd(A,b,x_0=None,x_tru=None,tol=10**-3,max_iter=500,recalc=50,full_output=False):
    """
    CG for symmetric, psd A with 1 matrix-vector multiplication per iteration
    """
    n = len(A)
    if x_0 is None:
        x_0 = np.random.randn(n)

    x = np.copy(x_0)
    i = 0
    r = b-np.dot(A,x)
    d = np.copy(r)
    del_new = np.dot(r,r)
    del_0 = np.copy(del_new)

    if full_output == True:
        resids = OrderedDict()
        start_time = time.time()
        resids[i] = la.norm(b-np.dot(A,x))
        if x_tru is not None:
            errs = OrderedDict()
            errs[i] = la.norm(x-x_tru)

    while not (i > max_iter or del_new < (tol**2)*del_0):

        q = np.dot(A,d)
        alpha = del_new / np.dot(d,q)
        x += alpha*d
        if i % recalc == 0:
            r = b-np.dot(A,x)
        else:
            r -= alpha*q

        ## updates
        del_old = np.copy(del_new)
        del_new = np.dot(r,r)
        beta = del_new / del_old
        d = r + beta*d
        i += 1

        if full_output == True:
            resids[i] = la.norm(b-np.dot(A,x))
            if x_tru is not None:
                errs[i] = la.norm(x-x_tru)

    if full_output == True:
        resids[i] = la.norm(b-np.dot(A,x))
        if i < max_iter:
            status = True
        else:
            status = False
        if x_tru is not None:
            errs[i] = la.norm(x-x_tru)
            return x, i, status, resids, errs
        else:
            return x, i, status, resids
    else:
        return x

# for any A
def conjugate_gradient(A, b, tol=0.001, x = None, numIter = 500, full_output=False):
    """
    Conjugate gradients on the normal equations.
    (Page 41 in "Painless Conjugate Gradient")

    A doesn't need to be symmetric, positive-definite, or even square.
    Use conjugate_gradient_ideal for matrices that satisfy the above conditions.
    """
    return conjugate_gradient_ideal(A = np.dot(A.T, A), \
                                    b = np.dot(A.T, b), x = x, \
                                    numIter = numIter, full_output=full_output)

# TO DO: BiCGStab

# This works, but much slower than CG for large/high condition number matrices
def iter_refinement_eps(A, b, tol=0.001, numIter=500, x=None, e=None, full_output=False):
    """
    Iterative refinement with epsilon smoothing.

    e: epsilon, value added to diagonal of A to lower condition number (decreases
                    w/ each iteration)
    """
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    n = len(ATA)
    if x is None:
        x = np.zeros(n)
    if e is None:
        e = 2*la.norm(ATA)

    #min_err = (np.copy(x), norm_dif(x, A, b))

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        e = 0.5 * e
        if np.random.uniform() < 0.01:
            print('IR iter %d; e: %f' % (i, e))


        r = ATb - np.dot(ATA, x)
        r_norm = la.norm(r)

        # break if residual blows up (becomes nan)
        if r_norm != r_norm:
            break

        if full_output:
            resids[time.time() - start_time] = r_norm

        # exit if close enough
        if r_norm < tol:
            if full_output:
                return x, i, True, resids
            else:
                return x

        A_e = np.copy(ATA) + e * np.identity(n)

        #print('ITER REFINE %d' % i)
        #d = gradient_descent_helper(np.copy(A_e), np.copy(r), np.copy(x))
        #d = conjugate_gradient(np.copy(A_e), np.copy(r), x=np.copy(x))


        x += np.dot(la.inv(A_e), r)

        #if norm_dif(x, A, b) < min_err[1]: min_err = (np.copy(x), norm_dif(x, A, b))


    #print('IR: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x



# ==============================================================================

def conjugate_gs_alt(U, A):
    """
    Conjugate Gram-Schmidt process.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    Args:
        (numpy.ndarray) U: array of n linearly independent column vectors.
        (numpy.ndarray) A: matrix for vectors to be mutually conjugate to.

    Returns:
        (numpy.ndarray) D: array of n mutually A-conjugate column vectors.
    """
    n = len(U)
    D = np.copy(U)
    beta = np.zeros([n,n])

    D[:, 0] = U[:, 0]
    for i in range(1, n):
        for j in range(0,i-1):

            Adj = np.dot(A, D[:, j])

            beta[i, j] = -np.dot(U[:, i].T, Adj)
            beta[i, j] /= np.dot(D[:, j].T, Adj) # (37)

            D[:, i] = U[:, i] + np.dot(beta[i, j], D[:, j]) # (36)

    ## checks
    for i in range(0, n-1):
        for j in range(i+1,n):
            # print( np.dot(U[:, i],np.dot(A,D[:, j])) + beta[i, j]*np.dot(D[:, j].T,np.dot(A,D[:, j])) )
            print( np.dot(D[:,i], np.dot(A, D[:,j])) )

    return D

def conjugate_gs(u, A):
    """
    Conjugate Gram-Schmidt process.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    Args:
        (numpy.ndarray) u: array of n linearly independent column vectors.
        (numpy.ndarray) A: matrix for vectors to be mutually conjugate to.

    Returns:
        (numpy.ndarray) d: array of n mutually A-conjugate column vectors.
    """
    n = len(u)
    d = np.copy(u)

    for i in range(1, n):
        for j in range(0,i):

            Adj = np.dot(A, d[:, j])


            Bij = -np.inner(u[:, i].T, Adj)
            Bij /= np.inner(d[:, j].T, Adj) # (37)

            d[:, i] += np.dot(Bij, d[:, j]) # (36)

    return d

def arnoldi(A,b):
    """
    Conjugate (modifed) Gram-Schmidt process for Krylov(A,b).

    Args:
        A:  matrix (psd)
        b:  vector RHS solution

    Returns:
        Q:  matrix (unitary/orthogonal) normalized vectors where
            Q[:,1], ... , Q[:,n] span K^n(A,b)
        H:  matrix (upper hessenberg) s.t.
            H = Q^T A Q upon completion
    """

    ## initialize
    n = len(A)
    Q = np.zeros([n,n])
    H = np.zeros([n,n])

    ## first vector
    Q[:,0] = b/la.norm(b)

    ## remaining vectors
    for j in range(n-1):               # start computation for Q[:,j+1]
        t = np.dot(A,Q[:,j])           # t \in K^[j+1](A,b)
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i],t)  # H[i,j] * Q[:,i] is proj of t onto Q[:,i]
            t -= H[i,j] * Q[:,i]       # remove proj (ORTHO)
        H[j+1,j] = la.norm(t)
        Q[:,j+1] = t/H[j+1,j]          # normalize (NORMALIZE)

    ## last column of H
    H[:,n-1] = np.dot(Q.T,np.dot(A,Q[:,n-1]))
    return Q,H

def test_arnoldi(A,b):
    Q,H = arnoldi(A,b)
    for i in range(len(Q)):
        for j in range(i):
            print(np.dot(Q[:,i],Q[:,j]))
    print(H - np.dot(Q.T,np.dot(A,Q)))

def jacobi(A,b,tol=0.001,maxiter=1000,x0=None):
    '''
    Solves Ax = b with Jacobi splitting method
        A \in R^[n x n]
        b,x \in R^n

    ONLY WORKS for matrices A such that spectral_radius(B) < 1, where
        B = D-1 E,
        D = diagonal elements of A (zero elsewhere),
        E = non-diagonal elements of A (zero on diagonal)

    '''

    n = A.shape[0]

    ## start
    if x0 == None:
        x0 = np.random.randn(n)

    ## construct matrix components
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i][i] = A[i][i]
    E = A-D
    Dinv = la.inv(D)
    B = np.dot(-Dinv,E)
    z = np.dot(Dinv,b)

    spec_rad = max(la.svd(B)[1])**2
    if spec_rad >= 1:
        print('Spectral radius of B (%f) >= 1. Method won\'t converge.' % spec_rad)
        print('Returning None.')
        return None
    else:
        print('Spectral radius of B: %f' % spec_rad)

    ## iterations
    x = x0
    for i in range(maxiter):
        x = np.dot(B,x) + z
        #print(la.norm(np.dot(A,x)-b))
        if la.norm(np.dot(A,x)-b) <= tol:
            break

    return x

def iter_refinement(A, b, tol=0.001, numIter=500, x=None, full_output=False):
    """
    Iterative refinement method.

    https://en.wikipedia.org/wiki/Iterative_refinement

    Works, but needs more testing on various sizes, condition numbers + initial
    error in Ax=b.
    """
    # tol *= la.norm(A)

    m = len(A)
    n = len(A.T)
    if x is None:
        x = np.zeros(n)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        #print('Iter %d' % i)

        # Compute the residual r
        r = b - np.dot(A, x)

        if full_output:
            resids[time.time() - start_time] = la.norm(r)

        # Solve the system (Ad = r) for d
        result = scopt.minimize(fun=norm_dif, x0=np.random.randn(m), \
                                args=(A, r), method='CG')
        d, success, msg = result.x, result.success, result.message
        # TODO: find out which method is best/quickest to solve this

        x += d


        if la.norm(b - np.dot(A, x)) < tol:
            print('IR: Close enough at iter %d' % i)
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    print('IR: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

# this doesn't work ever
def iter_refinement_const_eps(A, b, tol=0.001, numIter=500, x=None, e=None, full_output=False):
    assert len(A) == len(A.T)
    n = len(A)

    if e is None:
        print('MAGIC NUMEBR')
        e = la.norm(A) / 2.0 # MAGIC NUMBER
    if x is None:
        x = np.zeros(n)
    if full_output:
        resids = OrderedDict()
        st_time = time.time()

    A_eps = A + e*np.identity(n)

    for i in range(numIter):
        r = np.dot(A, x) - b
        r_norm = la.norm(r)

        if full_output:
            resids[time.time() - st_time] = r_norm

        if r_norm < tol:
            if full_output:
                return x, i, True, resids
            else:
                return x

        #d = conjugate_gradient(np.copy(A_eps), np.copy(r), x=np.copy(x))
        x -= np.dot(la.inv(A_eps), r)

    if full_output:
        return x, numIter, False, resids
    else:
        return x

# ==============================================================================
# GRADIENT DESCENT GRAVEYARD (?)

def gradient_descent_helper(A, b, x, alpha=0.01, tol=0.1, verbose=0):
    """
    Helper method for iter_refinement_eps (NOPE). Standard gradient descent that also
        works on non-symmetric matrices.
    """
    n_iter = 0
    start_time = time.time()

    while 1:
        #if np.random.uniform() <= 0.00001:
        if 1:
            print('n_iter: %d' % n_iter)
            print(norm_dif(x, A, b))

        err = np.dot(A, x) - b
        # check for nan
        if la.norm(err) == float('Inf'):
            print('something went horribly wrong in gradient_descent_helper')
            sys.exit(0)

        # return if close enough
        if la.norm(err) < tol:
            break

        gradient = np.dot(A.T, err) / len(A)
        # also return if not close enough, but gradient still ~= 0
        # (in case of overconstrained linear systems, for example)
        if la.norm(gradient) < 0.000001:
            break

        # update
        x -= alpha * gradient

        n_iter += 1


    if verbose:
        print('n_iter: %d' % n_iter)
        print('time: %f' % (time.time() - start_time))

    return x

# baseline; for symmetric, positive-definite A
def gradient_descent(A, b, tol=10**-5, x = None, numIter = 500, full_output=False):
    """
    Standard gradient descent for SYMMETRIC, POSITIVE-DEFINITE matrices.

    Re-calculates residual EVERY iteration (so slow but a bit more accurate).

    Args:
        numpy.ndarray A:    n x n transformation matrix.
        numpy.ndarray b:    n x 1 "target values".
        numpy.ndarray x:    n x 1 initial guess (optional).
        int     numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ab - x||.
    """
    n = len(A)
    if x is None: x = np.zeros(n)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    # Start descent
    for i in range(numIter):
        if full_output:
            resids[time.time() - start_time] = norm_dif(x, A, b)

        # ACTUAL ALGORITHM
        # ======================================================================
        # calculate residual (direction of steepest descent)
        r = b - np.dot(A, x)

        # calculate step size (via line search)
        a = np.inner(r.T, r) / float(np.inner(r.T, np.inner(A, r)))

        # update x
        x += a * r
        # ======================================================================

        if la.norm(b - np.dot(A, x)) < tol:
            print('GD: Close enough at iter %d' % i)
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    print('GD: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

# modification: 1 matrix-vector multiplication per iteration
def gradient_descent_alt(A, b, x0=None, x_tru=None, tol=10**-5, numIter=500, recalc=50, full_output=False):
    """
    Implementation of gradient descent for PSD matrices.

    Notes:
        Needs thorough testing.
        Only 1 matrix-vector computation is performed per iteration (vs 2).
        Slow history tracking.

    Args:
        (numpy.ndarray)     A:    n x n transformation matrix.
        (numpy.ndarray)     b:    n x 1 "target values".
        (numpy.ndarray)    x0:    n x 1 initial guess (optional).
        (numpy.ndarray) x_tru:    n x 1 true x (optional).
        (int)         numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ax - b||_2.
    """

    n = len(A)

    # Ensure sound inputs
    assert len(A.T) == n
    assert len(b) == n

    # Working with (n, ) vectors, not (n, 1)
    if len(b.shape) == 2: b = b.reshape(n, )
    if x0 is None:
        x0 = np.random.randn(n, )
    else:
        assert len(x0) == n
        if len(x0.shape) == 2: x0 = x0.reshape(n, ) # (n, ) over (n, 1)

    # diagnostics
    x_hist = []

    if full_output:
        resids = []

    # first descent step
    x = x0
    r_curr = b - np.dot(A, x)
    Ar_curr = np.dot(A,r_curr)
    a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))
    r_new = r_curr - a*Ar_curr
    x += a * r_curr

    if full_output:
        x_hist.append(x)
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        resids.append(err)

    # remaining descent steps
    for _ in range(1,numIter):

        # calculate residual (direction of steepest descent)
        r_curr = r_new

        # calculate step size (via analytic line search)
        Ar_curr = np.inner(A, r_curr)
        a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))

        # updates
        x += a * r_curr
        x_hist.append(x)

        # calculate residuals for next step
        if _ % recalc == 0:
            r_new = b - np.dot(A, x)
        else:
            r_new = r_curr - a*Ar_curr

        # add residuals
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        if full_output:
            resids.append(err)

        # stop if close
        if err < tol:
            print('GD_alt: Close enough at iter %d' % _)
            print(la.norm(r_new))
            if full_output:
                return x, _, True, resids
            else:
                return x

    print('GD_alt: Max iteration reached (%d)' % numIter)
    if full_output:
        return x, numIter, False, resids
    else:
        return x

# modifications: 1 matrix-vector multiplication per iteration; nonsymmetric (square) matrix A
def gradient_descent_nonsymm(A, b, x0=None, x_tru=None, tol=10**-5, numIter=500, recalc=50, full_output=False):
    """
    Implementation of gradient descent for nonsymmetric matrices (or symmetric, but slow in this case).

    Notes:
        Needs thorough testing; error BLOW UP
        Re-calculate residual EVERY iteration (so slow but a bit more accurate).
        Only 1 matrix-vector computation is performed per iteration (vs 2).
        Slow history tracking.

    Args:
        (numpy.ndarray)     A:    n x n transformation matrix.
        (numpy.ndarray)     b:    n x 1 "target values".
        (numpy.ndarray)    x0:    n x 1 initial guess (optional).
        (numpy.ndarray) x_tru:    n x 1 true x (optional).
        (int)         numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ax - b||_2.
    """

    n = len(A)

    # Ensure sound inputs
    assert len(A.T) == n
    assert len(b) == n

    # Working with (n, ) vectors, not (n, 1)
    if len(b.shape) == 2: b = b.reshape(n, )
    if x0 is None:
        x0 = np.random.randn(n, )
    else:
        assert len(x0) == n
        if len(x0.shape) == 2: x0 = x0.reshape(n, ) # (n, ) over (n, 1)

    # diagnostics
    x_hist = []

    if full_output:
        resids = []

    # first descent step
    x = x0
    AA = 1/2*A+A.T
    r_curr = b - np.dot(AA, x)
    Ar_curr = np.dot(AA,r_curr)
    a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))
    r_new = r_curr - a*Ar_curr
    x += a * r_curr

    if full_output:
        x_hist.append(x)
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        resids.append(err)

    # remaining descent steps
    for _ in range(1,numIter):

        # calculate residual (direction of steepest descent)
        r_curr = r_new

        # calculate step size (via analytic line search)
        AA = 1/2*A+A.T
        Ar_curr = np.inner(AA, r_curr)
        a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))

        # updates
        x += a * r_curr
        x_hist.append(x)

        # calculate residuals for next step
        if _ % recalc == 0:
            r_new = b - np.dot(AA, x)
        else:
            r_new = r_curr - a*Ar_curr

        # add residuals
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        if full_output:
            resids.append(err)

        # stop if close
        if err < tol:
            print('GD_alt: Close enough at iter %d' % _)
            print(la.norm(r_new))
            if full_output:
                return x, _, True, resids
            else:
                return x

    print('GD_alt: Max iteration reached (%d)' % numIter)
    if full_output:
        return x, numIter, False, resids
    else:
        return x

# ==============================================================================
