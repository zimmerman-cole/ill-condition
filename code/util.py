import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import optimize, visual, traceback, sys

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


def small_sing_vals(cond_num, m=50, n=50, max_sing=0.8):
    """
    Use to generate a matrix whose singular values all have magnitude less than
    one.

    NEED TO TEST THOROUGHLY.
    """
    if abs(max_sing) >= 1:
        print('Spectral radius of this matrix will have magnitude >= 1. Are you sure?')

    min_sing = max_sing / float(cond_num)
    s = np.array(sorted([np.random.uniform(low=min_sing, high=max_sing) for _ in range(min(m,n)-2)] + [min_sing, max_sing], reverse=True))

    A = np.random.randn(m, n)
    u,_,v = la.svd(A, full_matrices=False)

    # Sparse? instead of np.diag(s)
    return np.dot(u, np.dot(np.diag(s), v))

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
    num_mem = 100 # number of past commands to remember
    past_commands = []
    while True:
        try:
            sys.stdout.write('>>> ')
            inp = raw_input()
            if inp=='continue':
                break
            else:
                past_commands.append(inp)
                exec(inp)
        except KeyboardInterrupt:
            print('')
            break
        except BaseException:
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
    Returns SPD matrix A, solution x_true and RHS b (in Ax=b).
    """
    A = psd_from_cond(cond_num=cond_num, n=n)
    x_true = 4 * np.random.randn(n)
    b = np.dot(A, x_true)

    return A, b, x_true
