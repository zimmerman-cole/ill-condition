import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
import util, optimize
import tomo2D.drt as drt
import matplotlib.pyplot as plt

"""
Projection-based methods
    -POCS
    -RAAR
    -...
"""

def pocs(Kb, A, sb, lam, M, B=None, max_iter=500, tol=10**-5, full_output=0):
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
    max_iter:     Max number of iterations.
         tol:     Desired accuracy for minimization problem
                    (linear constraint must be completely accurate).

        full_output: TODO - for plotting intermediate info...

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

    # Set up solver for constraint term
    constr_solver = optimize.ConjugateGradientsSolver(
        A = (iden(n) - M.T.dot(M)).dot(A.T.dot(A) + lam * B.T.dot(B)), \
        b = np.zeros(n), full_output = 0
    )

    min_errors = []
    constr_errors = []

    u = np.zeros(n)

    for i in range(max_iter):
        print('=== Iter %d =============' % i)

        # === Solve minimization problem ================================
        u = min_solver.solve(x_0=u)
        min_err = la.norm(min_solver.A.dot(u) - min_solver.b)
        constr_err = la.norm(constr_solver.A.dot(u) - constr_solver.b)

        print('min err: %.2f' % min_err)
        print('constr err: %.2f\n' % constr_err)

        min_errors.append(min_err)
        constr_errors.append(constr_err)

        # === Solve constraint problem ==================================
        u = constr_solver.solve(x_0=u)
        min_err = la.norm(min_solver.A.dot(u) - min_solver.b)
        constr_err = la.norm(constr_solver.A.dot(u) - constr_solver.b)

        print('min err: %.2f' % min_err)
        print('constr err: %.2f' % constr_err)

        min_errors.append(min_err)
        constr_errors.append(constr_err)

        #raw_input()

    plt.plot(min_errors, marker='o')
    plt.plot(constr_errors, marker='o')
    plt.legend(['Minimization', 'Constraint'])
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error for each System')
    plt.show()

    return u

if __name__ == "__main__":
    # BLURRING PROBLEM
    if 1:
        # problem parameters
        m = 100     # number of pixels in image space
        n = m       # number of pixels in data space
        k = 10     # number of pixels in reconstruction ROI
        lam = 1.0   # lambda (reg. strength)

        # blur parameters
        sigma = 3
        t = 10

        # load 1d image
        filename = 'tomo1D/f_impulse_100.npy'
        sx = np.load(filename)

        print('Generating problem instance w/ params:')
        print('m: %d    k/p: %d   sig: %.2f   t: %d\n' % (m, k, sigma, t))
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



















    pass
