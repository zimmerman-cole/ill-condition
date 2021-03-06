import numpy as np
import numpy.linalg as la
from numpy.lib import scimath
import scipy.linalg as sla
import scipy.sparse.linalg as spsla
import scipy.sparse as sps
import matplotlib.pyplot as plt
import optimize, traceback, sys
from tomo1D import blur_1d as blur_1d
from tomo2D import blur_2d as blur_2d
import tomo2D.drt as drt
from cvxopt import spmatrix
import cvxpy as cvx

# not positive-definite
def mat_from_cond(cond_num, m=50, n=50, min_sing=None):
    """
    NOT POSITIVE-DEFINITE

    Generates an (m x n) matrix with the specified condition number. Use this
    to get a matrix with large (by most standards), decaying singular values.

    Args:
        (int)   cond_num:   Desired condition number.
        (int)          m:   Desired number of rows.
        (int)          n:   Desired number of columns.
        (float) min_sing:   Desired minimum singular value. Max singular value
                                will equal (cond_num * min_sing).

    Returns:
        Singular values of returned matrix will usually be large (depending
        on the supplied cond_num and min_sing, but usually the max >> 1).
        If you leave min_sing==None, it returns a ____NOEP____ matrix.
        If min_sing < 0, it returns a ____NOPE____ matrix. Singular value
        spectrum of returned matrix decreases roughly linearly (TODO: REPHRASE)
    """
    assert min(m,n) > 1
    if cond_num < 1:
        raise la.linAlgError('Condition number must be greater than or equal to 1')

    if min_sing is None:
        min_sing = abs(np.random.randn())

    max_sing = min_sing * float(cond_num)
    s = np.array(sorted([np.random.uniform(low=min_sing, high=max_sing) for _ in range(min(m,n)-2)] + [min_sing, max_sing], reverse=True))

    A = np.random.randn(m, n)
    u,_,v = la.svd(A, full_matrices=False)

    # Sparse? instead of np.diag(s)
    return np.dot(u, np.dot(np.diag(s), v))

def decaying_spd(cond_num, n=50, min_sing=None):
    """
    Gives a symmetric, positive-definite matrix with a 'bad' singular value
    spectrum shape (rapidly decaying values).
    """
    # =======================
    assert n > 1
    if cond_num < 1:
        raise la.linAlgError('Condition number must be greater than or equal to 1')
    if min_sing is None:
        min_sing = abs(np.random.randn())
    max_sing = min_sing * float(np.sqrt(cond_num))
    # =======================

    s = [max_sing] + [max_sing*(1.0/i) for i in range(1,n-1)] + [min_sing]
    for i in range(1, len(s)-1):
        # add some noise to the values
        s[i] += abs((0.1 * max_sing) * np.random.randn())

    A = np.random.randn(n, n)
    u,_,v = la.svd(A, full_matrices=False)

    B = np.dot(u, np.dot(np.diag(sorted(s)), v))
    # Sparse? instead of np.diag(s)
    return np.dot(B.T,B)

def hanging_spd(cond_num, n=50, pct_good=0.80, drop=0.90, min_sing=None):
    """
    Gives a symmetric, positive-definite matrix (SPD) with a 'good' singular value
    spectrum shape ('pct_good' slowly decaying values before a sharp drop off).
    """
    # =======================
    assert n > 1
    if cond_num < 1:
        raise la.linAlgError('Condition number must be greater than or equal to 1')
    if min_sing is None:
        min_sing = abs(np.random.randn())
    max_sing = min_sing * float(np.sqrt(cond_num))

    # =======================

    n_good = round(pct_good*n,0)
    n_bad = n - n_good
    # print(n_good)
    # print(n_bad)
    good = np.linspace(max_sing,drop*max_sing,n_good)
    if n_bad == 1:
        bad = [min_sing]
    else:
        bad = np.linspace(min_sing/drop,min_sing,n_bad)
    s = sorted(np.append(good,bad))

    for i in range(1, len(s)-1):
        # print(i)
        # add some noise to the values
        if i<= n_good:
            noise = ((1-drop)) * np.random.randn()
            s[i] = max(min(max_sing, s[i] + noise),min_sing)
            # print(s[i])
        else:
            noise = ((1-drop) * min_sing) *np.random.randn()
            s[i] = min(max(min_sing,abs(s[i] + noise)),max_sing)

    A = np.random.randn(n, n)
    u,_,v = la.svd(A, full_matrices=False)
    # print(s)
    # print(max_sing)
    # print(min_sing)
    B = np.dot(u, np.dot(np.diag(s), v))
    # Sparse? instead of np.diag(s)
    return np.dot(B.T,B)

def psd_from_cond(cond_num, n=50, min_sing=None):
    """
    Generates a square SYMMETRIC matrix with specified condition number. Use this
    to get a matrix with large (by most standards), decaying singular values.

    Args:
        (int)   cond_num:   Desired condition number.
        (int)          n:   Desired number of columns and columns.
        (float) min_sing:   Desired minimum singular value. Max singular value
                                will equal (cond_num * min_sing).

    Returns:
        Singular values of returned matrix will usually be large (depending
        on the supplied cond_num and min_sing, but usually the max >> 1).
        If you leave min_sing==None, it returns a positive-definite matrix.
        If min_sing < 0, it returns a negative-definite matrix.
    """
    assert n > 1
    if min_sing is None:
        min_sing = abs(np.random.randn())

    max_sing = min_sing * float(np.sqrt(cond_num))
    s = np.array(sorted([np.random.uniform(low=min_sing, high=max_sing) for _ in range(n-2)] + [min_sing, max_sing], reverse=True))

    A = np.random.randn(n, n)
    u,_,v = la.svd(A, full_matrices=False)

    B = np.dot(u, np.dot(np.diag(s), v))
    return np.dot(B.T,B)

def ghetto_command_line():
    """
    Unfinished
    """
    print

    while True:
        try:
            sys.stdout.write('>>> ')
            inp = raw_input()
            if inp=='continue':
                break
            else:
                exec(inp)

            sys.stdout.flush()
        except KeyboardInterrupt:
            print
            break
        except Exception:
            traceback.print_exc()

def iter_vs_cnum(solver, n_range=None, cnum_range=None, verbose=0, \
            construct_args=dict(), solve_args=dict() ):
    """
    Gather info on how many iterations it takes to fully solve Ax=b depending
        on A's size and condition number. A is always symmetric and
        positive-definite.

    Args:
        (optimize.Solver)   solver: Chosen solver class (NOT an instance of it).
        ([int])            n_range: Range of matrix sizes to test.
        ([int])         cnum_range: Range of condition numbers to test.
        (dict)      construct_args: Solver-specific CONSTRUCTOR parameters.
                                    (i.e. for IR General Solver, have to pass
                                    'intermediate_solver', 'intermediate_iter' ...)
        (dict)          solve_args: Solver-specific parameters to be passed to
                                    solver.solve(). (i.e. 'tol', 'eps' for IR ...)

    Returns:
        (dict)              results: Test results showing number of iterations
                                        required for convergence.

    More on 'results':
        Each key in 'results' corresponds to a size, and each corresponding
            value is a list of integers denoting the number of iterations until
            convergence (each of which corresponds to a condition number in
            cnum_range).

    Example:
    >>> solve_params = {'tol': 0.1}
    >>> construct_params = {'intermediate_solver': optimize.DirectInverseSolver}

    # Tests 9 total matrices (one 5x5 w/ cnum 10, one 5x5 w/ cnum 100 ...)
    >>> results = iter_vs_cnum(solver = optimize.IterativeRefinementGeneralSolver, \
                    n_range = [5, 50, 500, 5000], cnum_range = [10, 100, 1000], \
                    construct_args = construct_params, solve_args = solve_params)
    # 'results' is a dictionary of length 4 (for each size), and each value is a
    # list of length 3 (for each condition number)

    >>> results[50] # Show nums of iterations for (50 x 50) A w/ cnums 10,100,1000
    [40, 55, 82]


    """
    if n_range is None:
        n_range = range(50, 501, 50)
    if cnum_range is None:
        cnum_range = [10**i for i in range(1,10)]

    n_mats = len(n_range) * len(cnum_range)
    print('Evaluating %d matrices' % n_mats)

    results = dict()

    # Construct solver instance
    solver_ = solver(A=None, b=None, full_output=1, **construct_args)

    for n in n_range:
        results[n] = []
        plt.figure()
        plt.title('Size: %d by %d' % (n, n))
        for cond_num in cnum_range:
            if verbose: print('n: %d    cnum: %d' % (n, cond_num))

            #A = util.psd_from_cond(cond_num=cond_num, n=n)
            A = psd_from_cond(cond_num=cond_num, n=n)
            x_true = 4 * np.random.randn(n)
            b = np.dot(A, x_true)

            if verbose: print('Initial resid error: %f' % la.norm(b))


            solver_.A, solver_.b = A, b
            xopt, n_iter, resids, x_difs = solver_.solve(x_true=x_true, max_iter = n*2, \
                                                            **solve_args)
            results[n].append(n_iter)

            if verbose:
                print('Final resid error: %f' % la.norm(b - np.dot(A, xopt)))
                print('Took %d iter' % n_iter)

    return results

def gen_data(n=100, cond_num=100):
    """
    Returns SPD matrix A, solution x_true (box shape)
        and RHS b (in Ax=b).
    """
    A = psd_from_cond(cond_num=cond_num, n=n)
    x_true = np.array([100 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    b = np.dot(A, x_true)

    return A, b, x_true

## ========== Hotelling Observer Problem ==========
def gen_Kb(m=None, K_diag=None, sparse=True):
    """
    m: dimension of data space
    """
    if m is None:
        print("specify `m` in gen_Kb")
        sys.exit(0)

    if K_diag is None:
        d = abs(np.random.randn(m))
    else:
        d = K_diag

    if sparse:
        return sps.diags(diagonals=d)
    else:
        K = np.zeros([m,m])
        for i in range(m):
            K[i][i] = d[i]
        return K

def gen_M_1d(k=None, n=None, sparse=True):
    """
    centers the k-dim ROI in an n-vector
    Args
        k: dimension of ROI
        n: dimension of image space (number of 1d pixels)
    Returns
        M: a k x n matrix
    """
    if n is None:
        print("specify `n` in gen_M_1d")
        sys.exit(0)

    if k is None:
        print("specify `k` in gen_M_1d")
        sys.exit(0)

    ## find split1 and split 2 indices
    s1 = (n-k)/2
    s2 = n-(s1+k)

    ## build diagonal
    if sparse:
        data = np.ones(n)
        offsets = s1
        M = sps.dia_matrix((data,offsets), shape=(k,n))
    else:
        d = np.concatenate([np.zeros(s1), np.ones(k), np.zeros(s2)])
        M = np.diag(d)
        M = M[s1:(s1+k),:]
    return M

def gen_instance_1d_blur(m=None, n=None, k=None, K_diag=None, sigma=3, t=10, sparse=True):
    """
    Args
        m: dimension of data space
        k: dimension of ROI
        n: dimension of image space (number of 1d pixels)
        sigma: gaussian blur standard deviation
        t: gaussian blur pixel window size
    Returns
        M: a k x n matrix
    """
    Kb = gen_Kb(m=m, K_diag=K_diag, sparse=sparse)
    X = blur_1d.fwdblur_operator_1d(n=n, sigma=sigma, t=t, sparse=sparse)
    M = gen_M_1d(k=k, n=n, sparse=sparse)

    return Kb, X, M

def gen_M_2d(ri=None, k=None, n_1=None, n_2=None, sparse=True):
    """
    Generates a mask `M` to extract the middle `k` pixels from image row `ri`
    Args:
        ri: row index (beginning from zero) of interest in 2d image
        k: (centered) window length
        n_1: n rows of image
        n_2: n cols of image
    Returns:
        M: mask operator matrix
    """
    if ri is None:
        ri = int(float(n_1)/2.)

    t = np.zeros(n_1)
    t[ri] = 1
    M = t
    s1 = (n_2-k)/2
    for i in range(n_2-1):
        M = sla.block_diag(M,t)
    M = M[s1:(s1+k),:]
    if sparse:
        M = sps.csr_matrix(M)
    return M

def gen_instance_2d_blur(m=None, n_1=None, n_2=None, ri=None, k=None, K_diag=None, sigma=None, t=None, sparse=True):
    """
    Args
        m: dimension of data space
        n_1: num rows of image
        n_2: num cols of image
            --> n: (= n_1 * n_2) dimension of image space (number of 2d pixels)
        ri: row index for HO-ROI
        k: dimension of ROI
        sigma: gaussian blur standard deviation
        t: gaussian blur pixel window size
    Returns
        M: a k x n matrix
    """
    Kb = gen_Kb(m=m, K_diag=K_diag, sparse=sparse)
    X_col, X_row = blur_2d.fwdblur_operator_2d(n_1=n_1, n_2=n_2, sigma=sigma, t=t, sparse=sparse)
    X = X_col.dot(X_row)
    M = gen_M_2d(ri=ri, k=k, n_1=n_1, n_2=n_2, sparse=sparse)

    return Kb, X, M

def gen_instance_2d_xray(m=None, n_1=None, n_2=None, ri=None, k=None, K_diag=None, sparse=True):
    """
    Args
        m: dimension of data space
        n_1: num rows of image
        n_2: num cols of image
            --> n: (= n_1 * n_2) dimension of image space (number of 2d pixels)
        ri: row index for HO-ROI
        k: dimension of ROI
        sigma: gaussian blur standard deviation
        t: gaussian blur pixel window size
    Returns
        M: a k x n matrix
    """
    Kb = gen_Kb(m=m, K_diag=K_diag, sparse=sparse)
    X = drt.gen_X(n_1=n_1, n_2=n_2, m=m, sp_rep=sparse)
    M = gen_M_2d(ri=ri, k=k, n_1=n_1, n_2=n_2, sparse=sparse)

    return Kb, X, M

def scipy_sparse_to_spmatrix(A):
    """
    Takes scipy sparse matrix to a cvxopt spmatrix
    """
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def calc_hot(X=None, B=None, lam=None, M=None, u=None, ESI=False):
    m, n = X.shape[0], X.shape[1]
    if B is None: B = sps.eye(n)

    ## intermediate matrix
    Z = X.T.dot(X) + lam*B.T.dot(B)


    if ESI:
        w = M.dot(Z).dot(u[0:n])
        return w.reshape(len(w),1)
    else:
        w = M.dot(Z).dot(u)
        return w.reshape(len(w),1)

def direct_rxn(X=None, lam=None, B=None, sparse=True):
    n = X.shape[1]
    if B is None:
        if sparse:
            B = sps.eye(n)
        else:
            B = np.diag(np.ones(n))
    A = X.T.dot(X) + lam*B.T.dot(B)
    if sparse:
        R = spsla.spsolve(A, X.T, use_umfpack=True)
    else:
        R = la.solve(A, X.T)
    return R

def direct_solve(Kb=None, R=None, M=None, B=None, sb=None, sparse=True):
    MR = M.dot(R)
    Lx = MR.dot(Kb)
    Kx = Lx.dot(MR.T)
    sx = MR.dot(sb)
    if sparse:
        w = spsla.spsolve(Kx,sx)
        w = w.reshape(len(w),1)
    else:
        w = la.solve(Kx,sx)
        w = w.reshape(len(w),1)
    return w, Kx, sx

def gen_ESI_system(X=None, Kb=None, B=None, M=None, lam=None, sb=None):
    """
    Generates "Equivalent Symmetric Indefinite" LHS and RHS based on III
    """
    m, n = X.shape[0], X.shape[1]
    if B is None: B = sps.eye(n)

    ## intermediate calc
    Z = (X.T.dot(X) + lam*B.T.dot(B))

    ## block LHS
    A11 = X.T.dot(Kb).dot(X)
    A12 = Z.dot(sps.eye(n) - M.T.dot(M))
    A21 = A12.T
    # A22 = np.zeros([n,n])
    A = sps.bmat([[A11,A12], [A21,None]])

    ## block RHS
    b1 = X.T.dot(sb)
    b = np.concatenate([b1.reshape(n,), np.zeros(n).reshape(n,)])

    return A, b

def gen_ESI3_system(X=None, Kb=None, B=None, M=None, lam=None, sb=None, sparse=True, Kb_is_diag=True):
    """
    Generates "Equivalent Symmetric Indefinite" LHS and RHS based on III
    """
    m, n = X.shape[0], X.shape[1]
    if B is None: B = sps.eye(n)

    ## intermediate calc
    Z = (X.T.dot(X) + lam*B.T.dot(B))
    C = Z.dot(sps.eye(n) - M.T.dot(M))
    if sparse:
        if Kb_is_diag:
            K_12 = sps.spdiags([np.lib.scimath.sqrt(x) for x in Kb.diagonal()], diags=0, m=m, n=m)
        else:
            lu = spsla.splu(Kb)
            D_12 = sps.spdiags([np.lib.scimath.sqrt(x) for x in np.diag(lu.U.A)], diags=0, m=m, n=m)
            L = lu.L
            L_permuted = L.dot(D_12)
            Pr = sps.csc_matrix((n, n))
            Pc = sps.csc_matrix((n, n))
            Pr[lu.perm_r, np.arange(n)] = 1
            Pc[np.arange(n), lu.perm_c] = 1
            K_12 = Pr.T.dot(L_unarranged).dot(Pc.T)
        A22 = -sps.eye(m)
    else:
        if Kb_is_diag:
            K_12 = np.diag([np.lib.scimath.sqrt(x) for x in np.diag(Kb)], diags=0, m=m, n=m)
        else:
            K_12 = la.cholesky(Kb)
        A22 = -np.eye(m)

    ## intermediate calc
    Q = K_12.dot(X)

    ## block LHS
    A11 = None
    A12 = Q.T
    A13 = C.T
    A21 = Q
    # A22 definted above
    A23 = None
    A31 = C
    A32 = None
    A33 = None
    A = sps.bmat([[A11,A12,A13], [A21,A22,A23], [A31,A32,A33]])

    ## block RHS
    b1 = X.T.dot(sb)
    b = np.concatenate([b1.reshape(n,), np.zeros(m).reshape(m,), np.zeros(n).reshape(n,)])

    return A, b

def extend_ipm_prob(times=1, X=None, M=None, Kb=None, lam=None, sb=None, ZK=True):
    ## check
    assert(X is not None and M is not None and Kb is not None and lam is not None and sb is not None)

    ## size
    m = X.shape[0]
    n = X.shape[1]
    n_ext = n*times
    m_ext = m*times

    ## cleaning
    sb = sb.reshape(m,)

    ## extend X
    X_ext = sps.hstack([X for i in range(times)])
    X_ext = sps.vstack([X_ext for i in range(times)])

    ## extend M
    M_ext = sps.hstack([M for i in range(times)])
    M_ext = sps.vstack([M_ext for i in range(times)])

    ## extend Kb and sb
    Kb_diag = Kb.diagonal()
    Kb_diag_ext = np.zeros(m_ext)
    sb_ext = np.zeros(m_ext)
    for i in range(times):
        Kb_diag_ext[(i*m):((i+1)*m)] = Kb_diag
        sb_ext[(i*m):((i+1)*m)] = sb
    Kb_ext = sps.diags(Kb_diag_ext,0)

    ## cleaning
    sb_ext = sb_ext.reshape(m*times,1)

    ## compute Z and K cholesky
    if ZK:
        Z_ext = (sps.eye(n_ext) - M_ext.T.dot(M_ext)).dot(X_ext.T.dot(X_ext) + lam*sps.eye(n_ext))
        K_12_ext = sps.spdiags([np.sqrt(x) for x in Kb_diag_ext], diags=0, m=m_ext, n=m_ext)      # cholesky
        K_12_1_ext = sps.spdiags([1./x for x in K_12_ext.diagonal()], diags=0, m=m_ext, n=m_ext)  # inverse cholesky
        return X_ext, M_ext, Kb_ext, sb_ext, Z_ext, K_12_ext, K_12_1_ext, n_ext, m_ext
    else:
        return X_ext, M_ext, Kb_ext, sb_ext, n_ext, m_ext

def gen_ipm_prob(n=None, K_12=None, K_12_1=None, X=None, Z=None, sb=None, tol=None, niter=None):
    u = cvx.Variable(n)
    obj = cvx.Minimize( cvx.norm(K_12*X*u - K_12_1*sb))
    constr = [Z*u == float(0)]
    prob = cvx.Problem(obj, constr)

    mosek_params = {'MSK_IPAR_OPTIMIZER':2}  # 2: conic (http://docs.mosek.com/8.0/capi/constants.html#optimizertype)
    mosek_params['MSK_DPAR_INTPNT_CO_TOL_REL_GAP'] = tol      # doesn't seem to work with `tol` tolerance; breaks at 1e-14
    mosek_params['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = niter    # doesn't seem to work...
    mosek_params['MSK_IPAR_PRESOLVE_USE'] = 0 # setting MSK_PRESOLVE_MODE_OFF to off

    return prob, u
