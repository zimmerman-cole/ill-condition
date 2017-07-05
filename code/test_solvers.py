import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys, os
import time, datetime
import util, optimize
from scipy import optimize as scopt
from scipy.sparse import linalg as scla
from collections import OrderedDict

## ================== SIMULATION PARAMS ================== ##
#   n_sims:   (int):  number of simulations
# cond_num:   (int):  condition number
#        m:   (int):  rows of `A`
#        n:   (int):  cols of `A`
#   solver:   (str):  which solver type to run sims
## =================== PLOTTING PARAMS =================== ##
#    p_xax:  (bool):  x-axis option
#                       0: iteration, 1: time
#   p_comp:  (bool):  compare plots
#                       0: plot separately, 1: hold plots
## ==================== EXPORT PARAMS ==================== ##
#   e_name:   (str):  export directory name
#                       mkdir `e_name` under test_results
## ===================== EXAMPLE SIM ===================== ##
# t = Tester()
# t.fit(n_sims=3, cond_num=25, m=10, n=10, p_xax=0, p_comp=0)
# t.gen_data()
# s1 = "GradientDescentSolver"
# s2 = "ConjugateGradientsSolver"
# t.test_spsd(s1)
# t.test_spsd(s2)
## ======================================================= ##

class Tester:
    def __init__(self, \
                 n_sims=None, cond_num=None, m=None, n=None, solver=None, \
                 p_xax=None, \
                 e_name=None \
                ):

        self.n_sims, self.cond_num, self.m, self.n, self.solver = n_sims, cond_num, m, n, solver

        self.p_xax = p_xax

        self.e_name = e_name

    def __str__(self):
        l1 = 'Tester for %s Solver\n' % self.solver

        if self.n_sims is None: l2 = 'n_sims: None;\n'
        else: l2 = 'n_sims: %d;\n' % self.n_sims

        if self.cond_num is None: l3 = 'cond_num: None;\n'
        else: l3 = 'cond_num: %d;\n' % self.cond_num

        if self.m is None: l4 = 'm: None;\n'
        else: l4 = 'm: %d;\n' % self.m

        if self.n is None: l5 = 'n: None;\n'
        else: l5 = 'n: %d;\n' % self.n

        if self.p_xax is None: l6 = 'p_xax: None\n'
        else: l6 = 'p_xax: %s\n' % p_xax

        return l1+l2+l3+l4+l5+l6+l7

    def __repr__(self):
        return self.__str__()

    def fit(self, n_sims, cond_num, m, n, p_xax, p_comp):
        """
        Fit for random simulations
        """
        self.n_sims, self.cond_num, self.m, self.n = n_sims, cond_num, m, n
        self.A, self.b, self.x_0, self.x_true = None, None, None, None

        self.p_xax, self.p_comp = p_xax, p_comp

        fit_time = datetime.datetime.fromtimestamp( time.time() ).strftime('%Y-%m-%d_%H.%M.%S')
        self.e_name = "%s_%sx%s_%s" % (self.cond_num, self.m, self.n, fit_time)

    def gen_data(self):
        self.A, self.b, self.x_0, self.x_true = [], [], [], []
        for sim in range(self.n_sims):
            self.A.append( util.psd_from_cond(self.cond_num,self.n) )
            self.x_true.append( np.random.randn(self.n) )
            self.x_0.append( np.random.randn(self.n) )
            self.b.append( np.dot(self.A[sim],self.x_true[sim]) )

    def test_spsd(self,solver,**kwargs):
        """
        Test symmetric, psd matrices
        """
        ## error checking
        if self.n_sims is None or self.cond_num is None or (self.m is None and self.n is None):
            raise AttributeError('n_sims and/or cond_num and/or (m and n) haven\'t been set yet.')
        assert self.m == self.n
        solver_name = solver
        self.solver = eval("optimize."+solver_name)

        ## intermediate solver parameters
        if bool(kwargs) == True:
            intermediate_solver_name = kwargs["intermediate_solver"]

            self.intermediate_solver = eval("optimize."+intermediate_solver_name)
            self.intermediate_iter = kwargs["intermediate_iter"]
            self.intermediate_continuation = kwargs["intermediate_continuation"]

        ## initialize output objects
        x_opt_out = []
        i_out = []
        residuals_out = []
        errors_out = []
        path_out = []

        ## initialize plots
        fig_residuals = plt.figure("residuals")
        ax_residuals = plt.subplot(111)
        fig_errors = plt.figure("errors")
        ax_errors = plt.subplot(111)

        ## simulations
        for sim in range(self.n_sims):
            ## tracking
            print("==================== %s Simulation: %s ====================\n" % (solver_name, sim))
            ## error checking
            if self.A is None or self.b is None or self.x_0 is None or self.x_true is None:
                raise AttributeError('data (A, b, x_0, x_true) haven\'t been generated yet.')

            ## formulate problem instance
            A = np.copy(self.A[sim])
            x_true = np.copy(self.x_true[sim])
            x_0 = np.copy(self.x_0[sim])
            b = np.copy(self.b[sim])

            ## instantiate solver object
            if bool(kwargs) == False:
                solver_object = self.solver(A=A, b=b, full_output=True)
            else:
                solver_object = self.solver(A=A, b=b, full_output=True, \
                                            intermediate_solver = self.intermediate_solver, \
                                            intermediate_iter = self.intermediate_iter, \
                                            intermediate_continuation = self.intermediate_continuation \
                                            )

            ## solve simulated problem
            x_opt, i, residuals, errors = solver_object.solve(tol=10**-5, x_0=x_0, max_iter=500, recalc=20, x_true=x_true)
            path = solver_object.path(self, x_0=x_0, max_iter=500, recalc=20,  x_true=x_true)

            ## append output
            x_opt_out.append(x_opt)
            i_out.append(i)
            residuals_out.append(residuals)
            errors_out.append(errors)
            path_out.append(path)

            ## set axes
            if self.p_xax == 0:
                xax = range(len(residuals))
                xlab = "Iteration"
            else:
                xax = [x[1] for x in residuals]
                xlab = "Time"

            ## y vectors
            y_residuals = [x[0] for x in residuals]  # residuals
            ylab_residuals = "Residual ||Ax - b||"   # residuals
            y_errors = [x for x in errors]     # errors
            ylab_errors = "Error ||x - x_true||"     # errors

            ## plot residuals
            plt.figure("residuals")
            ax_residuals.plot(xax, y_residuals, label='sim-%s resids' % sim)
            plt.yscale('log')
            plt.xlabel(xlab)
            plt.ylabel(ylab_residuals)

            ## plot errors
            plt.figure("errors")
            ax_errors.plot(xax, y_errors, label='sim-%s errs' % sim)
            plt.yscale('log')
            plt.xlabel(xlab)
            plt.ylabel(ylab_errors)

        ## save plot(s)
        path_out = "../test_results/"+str(solver_name)+"/"+self.e_name
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        plt.figure("residuals")
        if bool(kwargs) == True:
            plt.title('%s Simulation Results (Intermediate: %s )' % (solver_name, intermediate_solver_name))
        else:
            plt.title('%s Simulation Results' % (solver_name))
        ax_residuals.legend(prop={'size':5})
        fig_residuals.savefig(path_out+'/residuals.png')

        plt.figure("errors")
        if bool(kwargs) == True:
            plt.title('%s Simulation Results (Intermediate: %s )' % (solver_name, intermediate_solver_name))
        else:
            plt.title('%s Simulation Results' % (solver_name))
        ax_errors.legend(prop={'size':5})
        fig_errors.savefig(path_out+'/errors.png')
        if self.p_comp == 0:
            fig_residuals.clf()
            ax_residuals.cla()
            fig_errors.clf()
            ax_errors.cla()

t = Tester()
t.fit(n_sims=3, cond_num=25, m=10, n=10, p_xax=0, p_comp=0)
t.gen_data()
s1 = "GradientDescentSolver"
s2 = "ConjugateGradientsSolver"
s3 = "IterativeRefinementGeneralSolver"
t.test_spsd(s1)
t.test_spsd(s2)
t.test_spsd(solver=s3,intermediate_solver="DirectInverseSolver",intermediate_iter=20,intermediate_continuation=True)
t.test_spsd(solver=s3,intermediate_solver="ConjugateGradientsSolver",intermediate_iter=20,intermediate_continuation=True)


# ==================== GRAVEYARD ==================== #

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

        x_0 = np.random.randn(n)
        x_arr, xopt = scla.cg(A, b, x_0=x_0)


        init_error = 100.0 * la.norm(x_0 - true_x) #/ float(la.norm(true_x))), 2)
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

        x_0 = np.random.randn(n)
        xopt, fopt, func_calls, grad_calls, warnflag = scopt.fmin_cg(f=norm_dif, x_0=x_0, args=(A, b), full_output=True)
        flags[warnflag] += 1

        init_error = 100.0 * la.norm(x_0 - true_x) #/ float(la.norm(true_x))), 2)
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
        x_0 = np.random.randn(n)
        xopt,info = scla.minres(A,b)


        init_error = la.norm(x_0 - true_x) #/ float(la.norm(true_x))), 2)
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
    """
    Plots residuals vs iteration number for:
        Conjugate Gradients
        Iterative Refinement w/ constant epsilon

    IR w/ const. epsilon doesn't work; residual just grows continually.
        (even when (A+Ieps) is inverted directly)
    """
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
    ir_results = optimize.iter_refinement_const_eps(A, b, numIter=n_iter, full_output=True)
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
