import numpy as np
import numpy.linalg as la
import traceback
import sys

# use as baseline to test conjugate gradient
def gradient_descent(X, g, f = None, numIter = 30):
    """
    Standard gradient descent for SYMMETRIC,
    POSITIVE-DEFINITE matrices.
    Needs thorough testing.

    Args:
        numpy.ndarray X:    n x n transformation matrix.
        numpy.ndarray g:    n x 1 "target values".
        numpy.ndarray f:    n x 1 initial guess (optional).
        int     numIter:    Number of passes over data.

    Returns:
        argmin(f) [Xf - g].
    """
    n = len(X)
    # Ensure sound inputs
    assert len(X.T) == n
    assert len(g) == n
    # Working with (n, ) vectors, not (n, 1)
    if len(g.shape) == 2: g = g.reshape(n, )
    if f is None:
        f = np.random.randn(n, )
    else:
        assert len(f) == n
        if len(f.shape) == 2: f = f.reshape(n, ) # (n, ) over (n, 1)

    # Start descent
    for _ in range(numIter):
        #print('Iter %d' % _)

        # calculate residual (direction of steepest descent)
        r = g - np.dot(X, f)
        if la.norm(r) < 0.000000000001: break

        # calculate step size (via line search)
        a = np.inner(r.T, r) / float(np.inner(r.T, np.inner(X, r)))

        # update x
        f += a * r

    return f

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



# REFERENCE IMPLEMENTATION
def conjugate_gradient_ideal(X, g, f = None, numIter = 30):
    """
    For SYMMETRIC, POSITIVE-DEFINITE matrices.
    """
    pass

def conjugate_gradient(X, g, f = None, numIter = 30):
    """
    placeholder
    """
    pass

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
    else: print('Spectral radius of B: %f' % spec_rad)

    ## iterations
    x = x0
    for i in range(maxiter):
        x = np.dot(B,x) + z
        #print(la.norm(np.dot(A,x)-b))
        if la.norm(np.dot(A,x)-b) <= tol:
            break

    return x
