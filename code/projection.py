import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import util, optimize, time
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

    ============================================================================
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
        assert sps.issparse(M)
    else:
        iden = np.identity
        assert not sps.issparse(M)

    # B default: identity
    if B is None: B = iden(n)


    # Set up solver for minimization term
    # A.T Kb A u = A.T sb
    min_solver = optimize.ConjugateGradientsSolver(
        A = A.T.dot(Kb.dot(A)), b = A.T.dot(sb), full_output=0
    )

    print('M.T M shape: ' + str(M.T.dot(M).shape))
    print('X.T X shape: ' + str(A.T.dot(A).shape))
    print('B.T B shape: ' + str(B.T.dot(B).shape))
    # Set up solver for constraint term [2]
    constr_solver = optimize.ConjugateGradientsSolver(
        A = (iden(n) - M.T.dot(M)).dot(A.T.dot(A) + lam * B.T.dot(B)), \
        b = np.zeros(n), full_output = 0
    )

    min_errors = []
    constr_errors = []

    u = np.zeros(n)

    for i in range(max_iter):
        #print('=== Iter %d =============' % i)

        # === Solve minimization problem ================================
        u = min_solver.solve(x_0=u)
        Au = np.array(min_solver.A.dot(u)).reshape(n, )
        min_err = la.norm(Au - min_solver.b)
        constr_err = la.norm(Au - constr_solver.b)

        #print('min err: %.2f' % min_err)
        #print('constr err: %.2f\n' % constr_err)

        min_errors.append(min_err)
        constr_errors.append(constr_err)

        # === Solve constraint problem ==================================
        u = constr_solver.solve(x_0=u)
        min_err = la.norm(min_solver.A.dot(u) - min_solver.b)
        constr_err = la.norm(constr_solver.A.dot(u) - constr_solver.b)

        #print('min err: %.2f' % min_err)
        #print('constr err: %.2f' % constr_err)

        min_errors.append(min_err)
        constr_errors.append(constr_err)

        if min_err <= tol:
            break

        #raw_input()

    if full_output:
        return u, min_errors, constr_errors
    else:
        return u

def raar(Kb, A, sb, lam, M, B=None, max_iter=500, tol=10**-5, full_output=0):
    """
    Relaxed Averaged Alternating Reflections.

    Solves the system:
                 min_u || Kb^.5 A u - Kb^-.5 sb || ^2

        s.t.    (I - M.T M)(A.T A + lam B.T B) u = 0
    ============================================================================
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
    raise NotImplementedError('')

def test(problem=0,method=1, plot=True):
    """
    Problem:    0 - blurring
                1 - small tomo problem  (10x10 square)
                2 - large tomo problem  (128x128 brain)

    Method:     0 - RAAR
                1 - POCS
    """

    lam = 1000.0

    mask_debug = False

    # BLURRING PROBLEM
    if problem == 0:
        # problem parameters
        n = 100     # number of pixels in image space
        m = n       # number of pixels in data space (same as img space)
        k = 20      # number of pixels in HO ROI
        # ^20 seems to be the minimum ROI size that has the system be solvable

        # blur parameters
        sigma = 3
        t = 10

        # load 1d image
        filename = 'tomo1D/f_impulse_100.npy'
        sx = np.load(filename)

        print('Generating blur problem w/ params:')
        print('m: %d    k/p: %d   sig: %.2f   t: %d\n' % (m, k, sigma, t))
        Kb, X, M = util.gen_instance_1d(m=m, n=n, k=k, \
                    K_diag=np.ones(m, dtype=np.float64), sigma=3, t=10, \
                    sparse=True)

        sb = X.dot(sx)

        print('Kb shape: ' + str(Kb.shape))
        print(' X shape: ' + str(X.shape))
        print(' M shape: ' + str(M.shape))
        print('sx shape: ' + str(sx.shape))
        print('sb shape: ' + str(sb.shape))
    # SMALL TOMO PROBLEM (10 x 10 square)
    elif problem == 1:

        # p and n are switched here compared to Sean's notes

        m = 10              # number of x-rays
        n_1, n_2 = 10, 10   # image dimensions
        n = n_1*n_2         # number of pixels (ROIrecon=full image)
        p = 2            # number of pixels in HO ROI
        # NOTE: ================================================================
        # ^11 seems to be the minimum ROI size that causes the system to be
        # able to be solved to a reasonable accuracy (~1 for 1000 iterations),
        # regardless of lambda
        # ===
        # For p < 11, the system seems to be unsolvable at high accuracies.
        # The lower the p, the farther apart the two systems/sets seem to be.
        # ---------------------------------
        # p value   |   error in min. term  (error in constraint term always=0)
        # ---------------------------------
        # 10        |   47.79
        # 9         |   49.90                       <==== ALSO m=10 for these
        # 8         |   64.26               NOTE: these are all for lambda=10.0.
        # 7         |   64.26                      higher lambdas will cause
        # 6         |   67.69                       slightly lower min. errors
        # ---------------------------------

        print('Generating tomo problem w/ params: ')
        print('m: %d    n: %d   lam: %d     p: %d' % (m, n, lam, p))

        # generate image (square)
        f = np.zeros((n_1, n_2))
        for i in range( int(0.3*n_1), int(0.7*n_1) ):
            for j in range( int(0.3*n_2), int(0.7*n_2) ):
                f[i, j] = 1
        f = f.reshape(n,)

        # generate forward projector X, sinogram g (sb)
        X = drt.gen_X(n_1=n_1, n_2=n_2, m=m, sp_rep=True)
        sb = X.dot(f)

        # generate covariance matrix (data space) and mask
        Kb = sps.eye(m)
        M = util.gen_M_1d(k=p, n=n, sparse=True)

        if mask_debug:
            _f_ = np.zeros((n_1, n_2))
            for i in range(n_1):
                for j in range(n_2):
                    _f_[i][j] = float(i)

            plt.figure()
            plt.title('Original image')
            plt.imshow(_f_)

            _f_ = _f_.reshape(n, )
            masked = M.dot(_f_).reshape(1, p)
            plt.figure()
            plt.title('Masked')
            plt.imshow(masked)
            plt.show()
            raise


        print(' X shape: ' + str(X.shape))
        print(' f shape: ' + str(f.shape))
        print('sb shape: ' + str(sb.shape))
        print('Kb shape: ' + str(Kb.shape))
        print(' M shape: ' + str(M.shape))
    # LARGER TOMO PROBLEM (128 x 128 brain)
    elif problem == 2:
        m = 100                 # number of x-rays
        n_1, n_2 = 128, 128     # image dimensions
        n = n_1*n_2             # number of pixels (ROIrecon=full image)
        p = 4500         # number of pixels in HO ROI
        # Works (min term converges ~> 0) for at least p=4096, but takes
        # forever (took 800.25 sec for p=8096)

        print('Generating tomo problem w/ params: ')
        print('m: %d    n: %d   lam: %d     p: %d' % (m, n, lam, p))

        # load image (brain)
        f = np.load('tomo2D/brain128.npy')
        f = np.array(f).reshape(n, )

        # generate forward projector X, sinogram g (sb)
        X = drt.gen_X(n_1=n_1, n_2=n_2, m=m, sp_rep=True)
        sb = X.dot(f)

        # generate covariance matrix (data space) and mask
        Kb = sps.eye(m)
        M = util.gen_M_1d(k=p, n=n, sparse=True)

        print(' X shape: ' + str(X.shape))
        print(' f shape: ' + str(f.shape))
        print('sb shape: ' + str(sb.shape))
        print('Kb shape: ' + str(Kb.shape))
        print(' M shape: ' + str(M.shape))
    else:
        raise ValueError('Possible problems: 0 (blur), 1 (small_tomo), 2 (large_tomo)')

    if method == 0:
        uopt = raar(Kb=Kb, A=X, sb=sb, lam=lam, M=M)
    elif method == 1:
        start_time = time.time()
        uopt, mins, cons = pocs(Kb=Kb, A=X, sb=sb, lam=lam, M=M, tol=1.0, \
            max_iter=10000, full_output=1)
        t = time.time() - start_time
        print('Took %.2f sec' % t)
        print('Final min err: %.2f' % mins[-1])

        if plot:
            plt.plot([mins[i] for i in range(len(mins)) if (i%2)], marker='o', markersize=3)
            # plt.plot(cons, marker='o', markersize=3)
            # plt.legend(['Minimization', 'Constraint'])
            plt.xlabel('Iteration')
            plt.ylabel('Absolute Error for each System')
            plt.show()

    else:
        raise ValueError('Choose from 1 (POCS) or 0 (RAAR)')


if __name__ == "__main__":

    test(problem=1, method=1, plot=True)























    pass