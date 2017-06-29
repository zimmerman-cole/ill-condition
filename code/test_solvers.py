import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time, sys, util, optimize, os
from scipy import optimize as scopt
from scipy.sparse import linalg as scla
from collections import OrderedDict


def norm_dif(x, *args):
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
        xopt, fopt, func_calls, grad_calls, warnflag = scopt.fmin_cg(f=norm_dif, x0=x0, args=(A, b), full_output=True)
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
        A = util.psd_from_cond(cond_num, n)

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

# Test CG ideal
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

# Test outdated iter. refinement
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
        x = optimize.iter_refinement(A, b, x=x)
        print('Final error: %f' % la.norm(true_x - x))
        print('Time taken: %f seconds' % (time.time() - st_time))

# DOESN'T WORK ANYMORE (DEPRECATED) (plots resids versus iter. num)
# TODO: FIX
def test_all_symmetric_pos_def(n, cond_num = 100, n_iter=100):
    """
    Test algorithms' performances on a SYMMETRIC, POSITIVE-DEFINITE matrix
        of given size and condition number.
    Old
    """

    print('Just use test_solvers.test_all() please')

    A = util.psd_from_cond(cond_num, n=n)
    true_x = 4 * np.random.randn(n) # 'magic' number
    b = np.dot(A, true_x)

    # Conjugate gradients
    cg_results = optimize.conjugate_gradient_ideal(A, b, numIter=n_iter, full_output=True)
    # Gradient descent
    gd_results = optimize.gradient_descent(A, b, numIter=n_iter, full_output=True)
    # Iterative refinement
    ir_results = optimize.iter_refinement(A, b, numIter=n_iter, full_output=True)

    plt.plot(cg_results[3], marker='o')
    plt.plot(gd_results[3], marker='o')
    plt.plot(ir_results[3], marker='o')


    plt.xlabel('Iteration')
    plt.ylabel('Residual ||Ax - b||')
    plt.yscale('log')
    plt.legend(['CG', 'GD', 'IR'])
    if type(cond_num) == float:
        plt.title('dim(A): %dx%d. cond(A): %.2f' % (n, n, round(cond_num, 2)))
    else:
        plt.title('dim(A): %dx%d. cond(A): %d' % (n, n, cond_num))
    plt.show()

# doesn't work
def test_iter_eps(n, e=100, cond_num = 100, n_iter=500):
    A = util.mat_from_cond(cond_num, m=n, n=n)
    true_x = 4 * np.random.randn(n)
    b = np.dot(A, true_x)

    print('Initial error: %f' % norm_dif(np.zeros(n), A, b))
    xopt, n_iter, suc, resids = optimize.iter_refinement_const_eps(A, b, e=e, full_output=True)
    print('Final error: %f (%d iter)' % (norm_dif(xopt, A, b), n_iter))

    plt.scatter(resids.keys(), resids.values(), marker='o')
    plt.ylabel('Residuals')
    plt.xlabel('Time')
    plt.yscale('log')
    #plt.show()

    return norm_dif(xopt, A, b)


# ==============================================================================
# RESIDUAL PLOTTING METHODS BELOW
# ==============================================================================

# TODO: make code below more readable

# Plot residuals vs iteration number
def test_all(m, n, cond_num = 100, n_iter = 100):
    A = util.mat_from_cond(cond_num=cond_num, m=m, n=n)
    true_x = 4 * np.random.randn(n) # 4 is 'magic' number
    b = np.dot(A, true_x)
    print('Initial absolute error: %f' % norm_dif(np.zeros(n), A, b) )

    # Conjugate gradients
    start = time.time()
    cg_results = optimize.conjugate_gradient(A, b, numIter=n_iter, full_output=True)
    print('CG took %f seconds' % (time.time() - start))
    print('CG final error: %f' % cg_results[3][next(reversed(cg_results[3])) ])

    # Iterative refinement
    start = time.time()
    ir_results = optimize.iter_refinement(A, b, numIter=n_iter, full_output=True)
    print('IR took %f seconds' % (time.time() - start))
    print('IR final error: %f' % ir_results[3][next(reversed(ir_results[3])) ])

    plt.plot(cg_results[3].values(), marker='o')
    plt.plot(ir_results[3].values(), marker='o')


    plt.xlabel('Iteration')
    plt.ylabel('Residual ||Ax - b||')
    plt.yscale('log')
    plt.legend(['CG', 'IR'])
    if type(cond_num) == float:
        plt.title('dim(A): %dx%d. cond(A): %.2f' % (n, n, round(cond_num, 2)))
    else:
        plt.title('dim(A): %dx%d. cond(A): %d' % (n, n, cond_num))
    plt.show()

# Plot residuals vs TIME at each iteration
def test_all_time(m, n, cond_num = 100, n_iter = 100):
    """
    TODO: study IR w/ epsilon smoothing stuff
    """

    A = util.mat_from_cond(cond_num=cond_num, m=m, n=n)
    true_x = 4 * np.random.randn(n) # 4 is 'magic' number
    b = np.dot(A, true_x)
    print('Initial absolute error: %f' % norm_dif(np.zeros(n), A, b) )

    # Conjugate gradients
    st_time = time.time()
    cg_results = optimize.conjugate_gradient(A, b, numIter=n_iter, full_output=True)
    print('CG final error: %f' % cg_results[3][next(reversed(cg_results[3]))] )
    print('CG took %f seconds (%d iter)' % (time.time() - st_time, cg_results[1]))

    # Iterative refinement w/ epsilon smoothing
    st_time = time.time()
    ir_results = optimize.iter_refinement_eps(A, b, numIter=n_iter, full_output=True)
    print('IR_eps final error: %f' % norm_dif(ir_results[0], A, b))
    print('IR_eps took %f seconds (%d iter)' % (time.time() - st_time, ir_results[1]))

    plt.scatter(cg_results[3].keys(), cg_results[3].values(), marker='x')
    plt.scatter(ir_results[3].keys(), ir_results[3].values(), marker='o')


    plt.xlabel('Time')
    plt.ylabel('Residual ||Ax - b||')
    plt.yscale('log')
    plt.legend(['CG', 'IR_eps'])
    if type(cond_num) == float:
        plt.title('dim(A): %dx%d. cond(A): %.2f' % (n, n, round(cond_num, 2)))
    else:
        plt.title('dim(A): %dx%d. cond(A): %d' % (n, n, cond_num))



    # num_mem = 100 # number of past commands to remember
    # past_commands = []
    # while True:
    #     try:
    #         sys.stdout.write('>>> ')
    #         inp = raw_input()
    #         if inp=='continue':
    #             break
    #         else:
    #             past_commands.append(inp)
    #             exec(inp)
    #     except KeyboardInterrupt:
    #         print('')
    #         break
    #     except BaseException:
    #         traceback.print_exc()
    plt.show()

# ==============================================================================
# LOAD/GATHER DATA
# ==============================================================================

# Gather data on weird iter_refine_eps
def gather_IR_data(m, n, cond_num = 100, n_iter = 100, n_mats=100):
    """
    Gather and save data for testing with iter_refinement_eps.
    Calling this method overwrites old data.
    """

    for i in range(n_mats):
        A = util.mat_from_cond(cond_num, m, n)
        true_x = 4 * np.random.randn(n)
        b = np.dot(A, true_x)
        init_err = norm_dif(np.zeros(n), A, b)
        print('Initial absolute error: %f' % init_err)

        st_time = time.time()
        res = optimize.iter_refinement_eps(A, b, full_output=True)
        print('Took %f seconds' % (time.time() - st_time))

        # A b; b is saved as column and appended onto A
        to_save = np.append(A, b.reshape(m, 1), axis=1)
        # ones that didn't blow up
        if norm_dif(res[0], A, b) <= 0.99 * init_err:
            path_ = '../test_results/iter_refine_eps/converges/m%d' % i
            np.save(path_, to_save)
        else:
            # ones that did
            path_ = '../test_results/iter_refine_eps/blows_up/m%d' % i
            np.save(path_, to_save)

def load_IR_data():

    b_path = '../test_results/iter_refine_eps/blows_up/'
    c_path = '../test_results/iter_refine_eps/converges/'

    b_filenames = [f for f in os.listdir(b_path) if f[-4:] == '.npy']
    c_filenames = [f for f in os.listdir(c_path) if f[-4:] == '.npy']

    b_arrs = []
    c_arrs = []

    for b in b_filenames:
        ar = np.load(b_path+b, 'r')
        b_arrs.append(ar)
    for c in c_filenames:
        ar = np.load(c_path+c, 'r')
        c_arrs.append(ar)

    return b_arrs, c_arrs














pass
