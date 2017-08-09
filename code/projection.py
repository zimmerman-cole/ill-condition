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

    u = np.zeros(n)

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

def dr(Kb, A, sb, lam, M, B=None, max_iter=500, tol=10**-5, full_output=0, order=None, sl=None):
    """
    Douglas-Rachford.

    Solves the system:
        [1] (min) min_u   || Kb^.5 A u - Kb^-.5 sb || ^2

        [2] (constr) s.t.    (I - M.T M)(A.T A + lam B.T B) u = 0

        with:   R_i = 2P_i - I  (for i = 1, 2)
                T_{1,2} = (1/2)(R_1 R_2 + I)
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
       order:     12 = T_12; 21 = T_21
          sl:     step length, default is 2 (i.e., reflection)

        full_output: TODO - for plotting intermediate info...

    Returns:
        Optimal u.
    """
    ## setup - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # shape
    n = A.shape[1]
    # sparsity
    if sps.issparse(A):
        iden = sps.eye
        assert sps.issparse(M)
    else:
        iden = np.identity
        assert not sps.issparse(M)
    # B default: identity
    if B is None: B = iden(n)
    # sl default: reflection
    if sl is None: sl = 2.

    ## operator - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # A.T Kb A u = A.T sb
    min_solver = optimize.ConjugateGradientsSolver(
        A = A.T.dot(Kb.dot(A)), b = A.T.dot(sb), full_output=0
    )
    # (I - M.T M)(A.T A + lam B.T B) u = 0
    constr_solver = optimize.ConjugateGradientsSolver(
        A = (iden(n) - M.T.dot(M)).dot(A.T.dot(A) + lam * B.T.dot(B)), \
        b = np.zeros(n), full_output = 0
    )

    print('M.T M shape: ' + str(M.T.dot(M).shape))
    print('X.T X shape: ' + str(A.T.dot(A).shape))
    print('B.T B shape: ' + str(B.T.dot(B).shape))

    min_errors = []         #
    constr_errors = []      #
    dr_min_errors = []      # dr min obj errors (before projection onto constraint)
    dr_constr_errors = []   # dr constraint errors (before projection onto constraint)
    proj_errors = []        # min errors after dr step projected onto constraint

    times = []

    t_0 = time.time()
    u_0 = np.zeros(n)

    for i in range(max_iter):
        print('=== Iter %d =============' % i)

        if order == 12:
            ## compute T_{1,2} - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ## first projection
            dd = constr_solver.solve(x_0=u_0)-u_0
            u_1 = u_0 + sl*dd
            constr_errors.append(la.norm(constr_solver.A.dot(u_1) - constr_solver.b))

            ## second projection
            v_0 = u_1
            d = min_solver.solve(x_0=v_0)-v_0
            v_1 = v_0 + sl*d
            min_errors.append(la.norm(min_solver.A.dot(v_1) - min_solver.b))

            ## average double projection with original position
            w_0 = 0.5*(u_0 + v_1)
            dr_min_errors.append(la.norm(min_solver.A.dot(w_0) - min_solver.b))
            dr_constr_errors.append(la.norm(constr_solver.A.dot(w_0) - constr_solver.b))

            ## project onto constraint - - - - - - - - - - - - - - - - - - - - - - -
            w_1 = constr_solver.solve(x_0=w_0)
            proj_errors.append(la.norm(min_solver.A.dot(w_1) - min_solver.b))

            ## update
            u_0 = w_0

            if proj_errors[-1] <= tol:
                break
        elif order == 21:
            ## compute T_{2,1} - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ## first projection
            dd = min_solver.solve(x_0=u_0)-u_0
            u_1 = u_0 + sl*dd
            min_errors.append(la.norm(min_solver.A.dot(u_1) - min_solver.b))

            ## second projection
            v_0 = u_1
            d = constr_solver.solve(x_0=v_0)-v_0
            v_1 = v_0 + sl*d
            constr_errors.append(la.norm(constr_solver.A.dot(v_1) - constr_solver.b))

            ## average double projection with original position
            w_0 = 0.5*(u_0 + v_1)
            dr_constr_errors.append(la.norm(constr_solver.A.dot(w_0) - constr_solver.b))
            dr_min_errors.append(la.norm(min_solver.A.dot(w_0) - min_solver.b))

            ## project onto constraint - - - - - - - - - - - - - - - - - - - - - - -
            w_1 = constr_solver.solve(x_0=w_0)
            proj_errors.append(la.norm(min_solver.A.dot(w_1) - min_solver.b))

            ## update
            u_0 = w_0

            if proj_errors[-1] <= tol:
                break
        times.append(time.time()-t_0)

    ## final project onto constraint - - - - - - - - - - - - - - - - - - - - - -
    w_0 = constr_solver.solve(x_0=w_0)
    proj_errors.append(la.norm(min_solver.A.dot(w_0) - min_solver.b))


    if full_output:
        return w_0, min_errors, constr_errors, dr_min_errors, dr_constr_errors, proj_errors, times
    else:
        return w_0


def raar(Kb, A, sb, lam, M, beta, B=None, max_iter=500, tol=10**-5, full_output=0):

    """
    Relaxed Averaged Alternating Reflections.
        (Wed. August 9)
        -Doesn't currently converge on our example 1D blurring problem for any
         value of beta.
        -Uses the T_(2,1) Douglas-Rachford instead of (1,2)
            (reflects first across P1, then P2)

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
    n = A.shape[1]

    if sps.issparse(A):
        iden = sps.eye
        assert sps.issparse(M)
    else:
        iden = np.identity
        assert not sps.issparse(M)

    # B default: identity
    if B is None: B = iden(n)

    assert 0.0 < beta and beta < 1.0

    u = np.zeros(n)

    # Set up solver for minimization term (P1)
    # A.T Kb A u = A.T sb
    min_solver = optimize.ConjugateGradientsSolver(
        A = A.T.dot(Kb.dot(A)), b = A.T.dot(sb), full_output=0
    )

    # Set up solver for constraint term [2] (P2)
    constr_solver = optimize.ConjugateGradientsSolver(
        A = (iden(n) - M.T.dot(M)).dot(A.T.dot(A) + lam * B.T.dot(B)), \
        b = np.zeros(n), full_output = 0
    )

    _min_errs_ = []
    _con_errs_ = []

    try:
        for i in range(max_iter):
            print('=== Iter %d =============' % i)

            # Calculate R_1 u ======================================================
            P1_u = min_solver.solve(x_0=np.copy(u))     # u projected onto P1
            R1_u = u + 2.0 * (P1_u - u)                 # u reflected across P1

            # Calculate R_2 R_1 u ==================================================
            P2_R1_u = constr_solver.solve(x_0 = np.copy(R1_u))
            R2_R1_u = R1_u + 2.0 * (P2_R1_u - R1_u)     # u reflected across P1, then P2

            # Take the average of the doubly-reflected u and original u for ========
            # Douglas-Rachford (2,1) operated u ====================================
            T21_u = 0.5 * (u + R2_R1_u)

            # Now use this to calculate RAAR-operated u ============================
            Vb_u = beta*T21_u + (1.0-beta)*P1_u

            # Now project onto P2 (ensure constraint is completely satisfied) ======
            p2_proj = constr_solver.solve(x_0 = Vb_u)

            # Check errors/termination condition ===================================
            min_err = la.norm(min_solver.A.dot(p2_proj) - min_solver.b)
            constr_err = la.norm(constr_solver.A.dot(p2_proj) - constr_solver.b)
            print('min err: %f' % min_err)
            print('constr err: %f' % constr_err)

            _min_errs_.append(min_err)
            _con_errs_.append(constr_err)

            u = p2_proj

            # P2 error=0 and P1 error <= tolerance
            if constr_err <= 10**-6 and min_err <= tol:
                break
    except KeyboardInterrupt:
        pass    # So you can interrupt the method and still plot the residuals so far


    print('============================================')
    print('FINAL min err: %.2f' % min_err)
    print('FINALconstr err: %.2f' % constr_err)

    if full_output:
        return u, _min_errs_, _con_errs_
    else:
        return u

def test(problem=0,method=1, plot=True):
    """
    Problem:    0 - small 1D blurring problem (100 pixels total)
                1 - small tomo problem  (10x10 square)
                2 - large tomo problem  (128x128 brain)
                3 - large 1D blurring problem (10000 pixels total)

    Method:     0 - RAAR
                1 - POCS
    """

    lam = 1000.0

    mask_debug = False

    # SMALL 1D BLURRING PROBLEM (100 pixels)
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
    # LARGER BLURRING PROBLEM (10000 pixels)
    elif problem == 3:
        # problem parameters
        n = 10000   # number of pixels in image space
        m = n       # number of pixels in data space (same as img space)
        k = 1999      # number of pixels in HO ROI
        # ^___ seems to be the minimum ROI size that has the system be solvable

        # blur parameters
        sigma = 3
        t = 10

        # load 1d image
        filename = 'tomo1D/f_impulse_10000.npy'
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
    else:
        raise ValueError('Possible problems: 0 (blur), 1 (small_tomo), 2 (large_tomo)')

    if method == 0:
        beta = 0.5
        print('RAAR method chosen; using beta=%.2f' % beta)
        start_time = time.time()
        uopt, mins, cons = raar(Kb=Kb, A=X, sb=sb, lam=lam, M=M, beta=beta, tol=1.0, \
            max_iter=10000, full_output=1)
        t = time.time() - start_time
        print('Took %.2f sec' % t)
        print('Final min err: %.2f' % mins[-1])

        if plot:
            plt.plot([mins[i] for i in range(len(mins)) if (i%2)], marker='o', markersize=3)
            plt.xlabel('Iteration')
            plt.ylabel('Absolute Error of Minimization Term')
            plt.show()

    elif method == 1:
        start_time = time.time()
        uopt, mins, cons = pocs(Kb=Kb, A=X, sb=sb, lam=lam, M=M, tol=1.0, \
            max_iter=10000, full_output=1)
        t = time.time() - start_time
        print('Took %.2f sec' % t)
        print('Final min err: %.2f' % mins[-1])

        if plot:
            plt.plot([mins[i] for i in range(len(mins)) if (i%2)], marker='o', markersize=3)
            plt.xlabel('Iteration')
            plt.ylabel('Absolute Error of Minimization Term')
            plt.show()
    elif method == 2:
        start_time = time.time()
        w_opt, mins, constrs, dr_mins, dr_constrs, projs, times = dr(Kb=Kb, A=X, sb=sb, lam=lam, M=M, tol=0.001, \
            max_iter=10000, full_output=1, order=21)
        t = time.time() - start_time
        print('Took %.2f sec' % t)
        print('Final min err: %.2f' % mins[-1])

        if plot:
            plt.loglog([mins[i] for i in range(len(mins)) if (i%2)], marker='o', markersize=10)
            plt.loglog(constrs, marker='o', markersize=10)
            plt.loglog(dr_mins, marker='o', markersize=6)
            plt.loglog(dr_constrs, marker='o', markersize=6)
            plt.loglog(projs, marker='o', markersize=3)
            plt.legend(['Minimization', 'Constraint', 'DR-mins', 'DR-constrs', 'Projs'])
            plt.xlabel('Iteration')
            plt.ylabel('Absolute Error for each System')
            plt.show()
    else:
        raise ValueError('Choose from 1 (POCS) or 0 (RAAR)')


if __name__ == "__main__":

    # test(problem=0, method=2, plot=True)
    test(problem=3, method=0, plot=True)






















    pass
