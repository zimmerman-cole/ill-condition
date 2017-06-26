import numpy as np
import numpy.linalg as la
import traceback, sys, scipy, time
from scipy import optimize as scopt
from collections import OrderedDict

# TODO: currently all (?) methods calculate residual an extra time each iteration
#               (when checking if resid norm is within tolerance range)

# || b - Ax ||
def norm_dif(x, *args):
    """
    Return || b - Ax || (Frobenius norm).
    """
    A, b = args
    return la.norm(b - np.dot(A, x))

# baseline; for symmetric, positive-definite A
def gradient_descent(A, b, tol=10**-5, x = None, numIter = 500, full_output=False):
    """
    Standard gradient descent for SYMMETRIC, POSITIVE-DEFINITE matrices.
    ## TODO: fix weird zigzag behaviour.

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

# TODO: add epsilon diagonal stuff
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

def iter_refinement_eps(A, b, tol=0.001, numIter=500, x=None, e=None, full_output=False):
    """
    Iterative refinement with epsilon smoothing.

    e: epsilon, value added to diagonal of A to lower condition number (decreases
                    w/ each iteration)
    """
    m, n = len(A), len(A.T)
    min_dim = min(m, n)
    if x is None:
        x = np.zeros(n)
    if e is None:
        e = la.cond(A) * 3

    min_err = (np.copy(x), norm_dif(x, A, b))

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(1,numIter+1):

        r = b - np.dot(A, x)

        # break if residual blows up (becomes nan)
        if la.norm(r) != la.norm(r):
            break

        if full_output:
            resids[time.time() - start_time] = la.norm(r)

        # exit if close enough
        if la.norm(r) < tol:
            if full_output:
                return min_err[0], i, True, resids
            else:
                return min_err[0]

        A_e = np.copy(A)
        A_e[:min_dim, :min_dim] += (e / (2.0*i)*np.identity(min_dim))

        d = conjugate_gradient(A_e, r, x)

        x += d

        if norm_dif(x, A, b) < min_err[1]: min_err = (np.copy(x), norm_dif(x, A, b))


    #print('IR: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return min_err[0], numIter, False, resids
    else:
        return min_err[0]


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

# ==============================================================================
# trash (we may find useful in the future somehow) below

# def gradient_descent_helper(A, b, x, alpha=0.01, tol=0.1, verbose=0):
#     """
#     Helper method for iter_refinement_eps (NOPE). Standard gradient descent that also
#         works on non-symmetric matrices.
#     """
#     n_iter = 1
#     start_time = time.time()
#
#     while 1:
#         n_iter += 1
#         if np.random.uniform() <= 0.00001:
#             print('n_iter: %d' % n_iter)
#             print(norm_dif(x, A, b))
#
#         err = np.dot(A, x) - b
#         # check for nan
#         if la.norm(err) != la.norm(err):
#             print('something went horribly wrong in gradient_descent_helper')
#             sys.exit(0)
#
#         # return if close enough
#         if la.norm(err) < tol:
#             break
#
#         gradient = np.dot(A.T, err) / len(A)
#         # also return if not close enough, but gradient still ~= 0
#         # (in case of overconstrained linear systems, for example)
#         if la.norm(gradient) < 0.000001:
#             break
#
#         # update
#         x -= alpha * gradient
#
#     if verbose:
#         print('n_iter: %d' % n_iter)
#         print('time: %f' % (time.time() - start_time))
#
#     return x
