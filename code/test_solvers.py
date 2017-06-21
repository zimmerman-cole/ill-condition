import numpy as np
import numpy.linalg as la
import time, sys
from scipy import optimize as scopt
from scipy.sparse import linalg as scla
import util, optimize

def func(x, *args):
    """
    Pass to scipy optimizer (to solve Ax=b):
        args[0]: A
        args[1]: b
    """
    A, b = args

    return la.norm(b - np.dot(A, x))

def test_cg_linear(cond_num, m, n, num_samples=20, verbose=0):
    """
    Test scipy.sparse.linalg.cg(...).
    DOESN'T CURRENTLY WORK.
    """
    st_time = time.time()
    improvements = []

    for _ in range(num_samples):
        A = util.mat_from_cond(cond_num, m, n)

        # Randomly generate true x.
        true_x = np.random.randn(n)
        b = np.dot(A, true_x)

        x0 = np.random.randn(n)
        x_arr, xopt = scla.cg(A, b, x0=x0)


        init_error = 100.0 * la.norm(x0 - true_x) #/ float(la.norm(true_x))), 2)
        if verbose >= 2: print
        opt_error = 100.0 * la.norm(xopt - true_x) #/ float(la.norm(la.norm(true_x)))), 2)

        improvements.append(init_error - opt_error)
        if verbose >= 2:
            print('==== Sample %d ==============================' % _)

            print('Initial percent error: %f' % init_error)
            print('Final (optimized) percent error: %f' % opt_error)
            print('')

    avg_time = (time.time() - st_time) / float(num_samples)
    avg_improv = round((sum(improvements) / float(num_samples)), 2)

    if verbose >= 1:
        print('Processing %d samples in which A has dimensions (%d by %d)' % (num_samples, m, n))
        print('Average time to solve system one time: %f seconds.' % avg_time)
        print('Average percent improvement: %f%%' % avg_improv)

    return avg_time, avg_improv

def test_cg_nonlinear(cond_num, m, n, num_samples=20, verbose=0):
    """
    Test accuracy of scipy.optimize.fmin_cg on a matrices of a given
    condition number.

    Args:
        (int)    cond_num:   Desired condition number.
        (int)           m:   Desired number of rows.
        (int)           n:   Desired number of columns.
        (int) num_samples:   Desired number of distinct matrices with condition
                                number cond_num to test fmin_cg on.

    Returns:
        (float) average time required to solve one Ax=b.
        (float) average percent error improvement (lowering ||b - Ax|| / ||b||)
    """

    improvements = []
    flags = dict()
    flags[0] = 0
    flags[1] = 0
    flags[2] = 0

    st_time = time.time()


    for _ in range(num_samples):
        A = util.mat_from_cond(cond_num, m, n)

        # Randomly generate true x.
        true_x = np.random.randn(n)
        b = np.dot(A, true_x)

        x0 = np.random.randn(n)
        xopt, fopt, func_calls, grad_calls, warnflag = scopt.fmin_cg(f=func, x0=x0, args=(A, b), full_output=True)
        flags[warnflag] += 1

        init_error = 100.0 * la.norm(x0 - true_x) #/ float(la.norm(true_x))), 2)
        opt_error = 100.0 * la.norm(xopt - true_x) #/ float(la.norm(la.norm(true_x)))), 2)

        improvements.append(init_error - opt_error)
        if verbose >= 2:
            print('==== Sample %d ==============================' % _)

            print('Initial percent error: %f' % init_error)
            print('Final (optimized) percent error: %f' % opt_error)
            print('')

    avg_time = (time.time() - st_time) / float(num_samples)
    avg_improv = round((sum(improvements) / float(num_samples)), 2)

    if verbose >= 1:
        print('Processing %d samples in which A has dimensions (%d by %d)' % (num_samples, m, n))
        print('Average time to solve system one time: %f seconds.' % avg_time)
        print('Average percent improvement: %f%%' % avg_improv)
        print('0: %d   1: %d    2: %d' % (flags[0], flags[1], flags[2]))

    return avg_time, avg_improv

def test_minres(cond_num, n, shift, num_samples=20, verbose=0):
    """
    Test accuracy of scla.minres optimization function on symmetric PSD matricies
    with specified condition number.

    Args:
        (int)    cond_num:   Desired condition number.
        (int)           m:   Desired number of rows.
        (int)           n:   Desired number of columns.
        (int) num_samples:   Desired number of distinct matrices with condition
                                number cond_num to test minres.
        (float)     shift:   Desired regularization shift.

    Returns:
        (float) average time required to solve one Ax=b.
        (float) average percent error improvement (lowering ||b - Ax|| / ||b||)

    Examples:
        (1)       no shift:   test_minres(cond_num=20, n=50, shift = 0, num_samples=20, verbose=2)
        (2)      yes shift:   test_minres(cond_num=20, n=50, shift = -3, num_samples=20, verbose=2)
    """

    abs_improvements = []  # absolute improvement (reduction) in absolute error
    pct_improvements = []  # %-age improvement (reduction) in absolute error

    st_time = time.time()

    for _ in range(num_samples):
        # randomly generate A
        A = util.psd_with_cond(cond_num, n)

        # randomly generate true x.
        true_x = np.random.randn(n)
        b = np.dot(A, true_x)

        # randomly generate initial start
        x0 = np.random.randn(n)
        xopt,info = scla.minres(A,b)


        init_error = la.norm(x0 - true_x) #/ float(la.norm(true_x))), 2)
        opt_error = la.norm(xopt - true_x) #/ float(la.norm(la.norm(true_x)))), 2)
        pct_error = 100*(init_error - opt_error)/init_error

        abs_improvements.append(init_error - opt_error)
        pct_improvements.append(100*abs_improvements[_]/init_error)

        if verbose >= 2:
            print('==== Sample %d ==============================' % _)
            print('Condition number of A: %f' % la.cond(A))
            print('Initial absolute error: %f' % init_error)
            print('Final (optimized) absolute error: %f' % opt_error)
            print('Percent reduction in abs error: %f' % pct_error)
            print('')

    avg_time = (time.time() - st_time) / float(num_samples)
    avg_abs_improv = round((sum(abs_improvements) / float(num_samples)), 2)
    avg_pct_improv = round((sum(pct_improvements) / float(num_samples)), 2)

    if verbose >= 1:
        print('Processing %d samples in which A has dimensions (%d by %d)' % (num_samples, n, n))
        print('Average time to solve system one time: %f seconds.' % avg_time)
        print('Average absolute improvement: %f' % avg_abs_improv)
        print('Average percent improvement: %f%%' % avg_pct_improv)
        # print('0: %d   1: %d    2: %d' % (flags[0], flags[1], flags[2]))

    return avg_time, avg_pct_improv

def test_cg(n, cond_num=50, tol=0.000001, num_samples=20, verbose=0):
    """
    Test optimize.conjugate_gradient_ideal(...).
    Behaviour is as expected - systems with higher
    condition numbers take longer to solve accurately.
    """

    # Average number of iterations needed to solve system to
    # specified tolerance
    avg_iter = 0

    for _ in range(num_samples):
        sys.stdout.write('Sample %d: ' % (_+1))
        # symmetric, positive-definite
        A = util.mat_from_cond(cond_num, m=n, n=n)
        A = np.dot(A, A.T)

        true_x = np.random.randn(n) + 5
        b = np.dot(A, true_x)

        x = np.random.randn(n)

        if verbose >= 1: print('Initial error: %f' % la.norm(true_x - x))
        x, n_iter, success = optimize.conjugate_gradient_ideal(A, b, tol=0.000001, x=x, full_output=True)
        if verbose >= 1: print('Final error: %f' % la.norm(true_x - x))

        avg_iter += n_iter

    avg_iter /= float(num_samples)

    print('Average n_iter needed for condition number ~%d: %f' % (cond_num, avg_iter))

def test_iter_refine(m, n, cond_num = 25, num_samples=20):
    """
    Test optimize.iter_refinement(...).
    UNFINISHED TESTING.
    """


    for _ in range(num_samples):
        print('Sample %d =========' % _)
        A = util.mat_from_cond(cond_num, m=m, n=n)
        true_x = np.random.randn(n) + 5
        b = np.dot(A, true_x)

        x = np.random.randn(n) + 4

        st_time = time.time()
        print('Initial error: %f' % la.norm(true_x - x))
        x = optimize.iter_refinement(A, b, x0=x, debug=0)
        print('Final error: %f' % la.norm(true_x - x))
        print('Time taken: %f seconds' % (time.time() - st_time))
