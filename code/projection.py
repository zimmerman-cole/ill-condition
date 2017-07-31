import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
import util, optimize

"""
Projection-based methods
    -POCS
    -RAAR
    -...
"""

def pocs(Kb, A, sb, lam, M, B=None, full_output=0):
    """
    Projection onto Convex Sets.

    Alternating projection method to solve system:
        min_u || Kb^.5 A u - Kb^-.5 sb || ^2

    Subject to one of the two (equivalent) constraints:
        [1]: (A.T A + lam B.T B) u = M.T w          <= not this one
        [2]: (I - M.T M)(A.T A + lam B.T B) u = 0

    Uses Conjugate Gradient method to solve the systems at each
        iteration.

    Args:
          Kb:     Covariance matrix (in data space).
           A:     Forward projector/blurrer.
          sb:     Signal in data space.
         lam:     Regularization strength.
           M:     Mask matrix.
           B:     Regularization matrix (i.e. identity or
                        finite differencing).

      full_output: for plotting intermediate info...
                    TODO

    Returns:
        Optimal u.
    """
    n = A.shape[1]

    if sps.issparse(A):
        iden = sps.eye
    else:
        iden = np.identity

    # B default: identity
    if B is None: B = iden(n)


    # Set up solver for minimization term
    # A.T Kb A u = A.T sb
    min_solver = optimize.ConjugateGradientsSolver(
        A = A.T.dot(Kb.dot(A)), b = A.T.dot(sb), full_output=0
    )
    print('min mat: ' + str(min_solver.A.shape))
    print('  min b: ' + str(min_solver.b.shape))

    # Set up solver for constraint term
    constr_solver = optimize.ConjugateGradientsSolver(
        A = (iden(n) - M.T.dot(M)).dot(A.T.dot(A) + lam * B.T.dot(B)), \
        b = np.zeros(n), full_output = 0
    )

    u = np.zeros(n)

    for i in range(100):
        print('=== Iter %d =============' % i)
        u = min_solver.solve(x_0=u)
        print('min err: %.2f' % la.norm(min_solver.A.dot(u) - min_solver.b))
        print('constr err: %.2f\n' % la.norm(constr_solver.A.dot(u) - constr_solver.b))

        u = constr_solver.solve(x_0=u)

        print('min err: %.2f' % la.norm(min_solver.A.dot(u) - min_solver.b))
        print('constr err: %.2f' % la.norm(constr_solver.A.dot(u) - constr_solver.b))

        raw_input()

if __name__ == "__main__":
    # problem parameters
    m = 100     # number of pixels in image space
    n = m       # number of pixels in data space
    k = 100     # number of pixels in ROI
    lam = 1.0   # lambda (reg. strength)

    # blur parameters
    sigma = 3
    t = 10

    # load 1d image
    filename = 'tomo1D/f_impulse_100.npy'
    sx = np.load(filename)

    print('Generating problem instance w/ params:')
    print('m: %d    k: %d   sig: %.2f   t: %d\n' % (m, k, sigma, t))
    Kb, X, M = util.gen_instance_1d(m=m, n=n, k=k, \
                K_diag=np.ones(m, dtype=np.float64), sigma=3, t=1, \
                sparse=True)

    sb = X.dot(sx)

    print('Kb shape: ' + str(Kb.shape))
    print(' X shape: ' + str(X.shape))
    print(' M shape: ' + str(M.shape))
    print('sx shape: ' + str(sx.shape))
    print('sb shape: ' + str(sb.shape))

    uopt = pocs(Kb=Kb, A=X, sb=sb, lam=lam, M=M)
