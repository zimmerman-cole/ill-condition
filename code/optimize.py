import numpy as np
import numpy.linalg as la

# use as baseline to test conjugate gradient
def gradient_descent(X, g, f = None, numIter = 30):
    """
    Standard gradient descent for SYMMETRIC,
    POSITIVE-DEFINITE matrices.
    Needs thorough testing.

    Args:
        numpy.ndarray X: n x n transformation matrix.
        numpy.ndarray g: n x 1 "target values".
        numpy.ndarray f: n x 1 initial guess (optional).
        int     numIter: Number of passes over data.

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
