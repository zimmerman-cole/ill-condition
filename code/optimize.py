import numpy as np
import numpy.linalg as la
import traceback, sys, scipy, time, sklearn
from scipy import optimize as scopt
from collections import OrderedDict
from sklearn.linear_model import SGDClassifier


class Solver:
    """
    Parent class for linear solvers.
    """

    def __init__(self, A=None, b=None, full_output=False, **kwargs):
        ## data input/output parameters
        self.A, self.b = A, b
        self.full_output = full_output

        ## intermediate solver parameters
        if bool(kwargs) == True:
            self.intermediate_solver = kwargs["intermediate_solver"]
            self.intermediate_iter = kwargs["intermediate_iter"]
            self.intermediate_continuation = kwargs["intermediate_continuation"]

    def _check_ready(self):
        """
        Check everything's in place to start optimization. DOES NOT
        check A is symmetric, positive-definite, or even square.

        Should preferably be overridden by child solver.
        """
        if self.A is None or self.b is None:
            raise AttributeError('A and/or b haven\'t been set yet.')
        if len(self.A) != len(self.b):
            raise la.LinAlgError('A\'s dimensions do not line up with b\'s.')


    def solve(self, tol=10**-5, x_0 = None, max_iter = 500, x_true = None, **kwargs):
        """
        Solve the linear system Ax = b for x.

        Args:
            (number)         tol:   Desired degree of accuracy (||residual|| <= tol).
            (np.array)        x_0:   Initial guess for x.
            (int)       max_iter:   Maximum number of iterations.
            (np.array)    x_true:   True solution to system. If provided (and
                                      full_outFput=True), the solver tracks
                                      ||x - x_true|| at each iteration.
                          kwargs:   Solver-specific parameters, e.g.
                                        -'eps' for iterative refinement
                                        -'recalc' for GD/CG

        Returns:
            If full_output=True and x_true is provided:
                (np.array)                   x:   The approximated solution.
                (int)                   n_iter:   Number of iterations taken.
                ([(float, float)])   residuals:   For each iteration, a tuple
                                                    containing the size of the
                                                    current residual and time elapsed
                                                    so far.
                ([float])               x_difs:   ||x - x_true|| at each iteration.

            If full_output=True and x_true is not provided, only x, n_iter and
                residuals are returned.

            If full_output=False, just x is returned.
        """
        self._check_ready()
        if x_0 is None:
            x = np.random.randn(len(self.A))
        else:
            x = np.copy(x_0)

        if self.full_output:
            return self._full(tol, x, max_iter, x_true, **kwargs)
        else:
            return self._bare(tol, x, max_iter, **kwargs)

    def _full(*args, **kwargs):
        print('_full not implemented?')

    def _bare(*args, **kwargs):
        print('_bare not implemented?')

    def test_methods(self):
        """
        Make sure _full, _bare and path give roughly the same x.
        """
        x_0 = np.random.randn(len(self.A))
        x_full = self._full(tol=10**-5, x=np.copy(x_0),   max_iter=500, recalc=20, x_true=None)[0]
        x_bare = self._bare(tol=10**-5, x=np.copy(x_0),   max_iter=500, recalc=20)
        x_path = self.path(tol=10**-5,  x_0=np.copy(x_0), max_iter=500, recalc=20)[-1]

        print('Full-bare: %f' % la.norm(x_full - x_bare))
        print('Full-path: %f' % la.norm(x_full - x_path))
        print('Bare-path: %f' % la.norm(x_bare - x_path))

class DirectInverseSolver(Solver):

    def __str__(self):
        l1 = 'Direct Inverse Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def __repr__(self):
        return self.__str__()

    def _full(self, tol, x, max_iter, x_true, **kwargs):

        ## initialize
        i = 0
        start_time = time.time()
        residuals = []
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]

        ## residuals (0)
        r = self.b - np.dot(self.A, x)
        r_norm = la.norm(r)
        residuals.append((r_norm, time.time() - start_time))

        ## solve
        x = la.solve(self.A,self.b)

        ## residuals (1)
        r = self.b - np.dot(self.A, x)
        r_norm = la.norm(r)
        residuals.append((r_norm, time.time() - start_time))

        if x_true is not None:
            x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals
        else:
            return x, i, residuals, x_difs

    def _bare(self, tol, x, max_iter, **kwargs):
        x = la.solve(self.A,self.b)
        return x

    def path(self, tol=10**-5, x_0=None, max_iter = 500, **kwargs):
        path = [x]
        x = la.solve(self.A,self.b)
        path.append(x)
        return path

# || b - Ax ||
def norm_dif(x, *args):
    """
    Return || b - Ax || (Frobenius norm).
    """
    A, b = args
    return la.norm(b - np.dot(A, x))

class GradientDescentSolver(Solver):
    """
    Gradient descent solver. Only works for SYMMETRIC, POSITIVE-DEFINITE matrices.

    Extra parameter(s) for Solver.solve(...):
        (int) recalc:   Directly recalculate the residual b - Ax every 'recalc'
                            iterations.
    """

    def __str__(self):
        l1 = 'Gradient Descent Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def __repr__(self):
        return self.__str__()

    def _full(self, tol, x, max_iter, x_true, **kwargs):
        """
        Tracks everything (times/iteration, residuals, etc.).

        If you provide an x_true, it also tracks ||x - x_true|| at each
            iteration.
        """
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])

        start_time = time.time()
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]

        # First descent step ======================================
        r = self.b - np.dot(self.A, x)
        r_norm = la.norm(r)
        residuals = [(r_norm, time.time() - start_time)]

        # Check if close enough already
        if r_norm <= tol:
            if x_true is None:
                return x, 0, residuals
            else:
                return x, 0, residuals, x_difs

        # If not, take a step
        i = 1
        Ar = np.dot(self.A, r)
        a = np.inner(r.T, r) / np.dot(r.T, Ar)
        x += a * r
        if x_true is not None:
            x_difs.append(la.norm(x - x_true))
        # =========================================================

        # Rest of descent
        while i < max_iter:
            # Directly calculate residual every 'recalc' steps
            if (i % recalc) == 0:
                r = self.b - np.dot(self.A, x)
            else:
                # Else, update using one less matrix-vector product
                r -= a * Ar
            r_norm = la.norm(r)
            residuals.append((r_norm, time.time() - start_time))

            # Check if close enough
            if r_norm <= tol: break
            i += 1

            # If not, take another step
            Ar = np.dot(self.A, r)
            x += a * r
            if x_true is not None:
                x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals
        else:
            return x, i, residuals, x_difs

    def _bare(self, tol, x, max_iter, **kwargs):
        """
        For max performance.
        """
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])

        # First descent step ==========================================
        r = self.b - np.dot(self.A, x)
        # Check if close enough already
        if la.norm(r) <= tol:
            return x

        # If not, take a step
        Ar = np.dot(self.A, r)
        a = np.inner(r.T, r) / np.dot(r.T, Ar)
        x += a * r
        # ==============================================================

        for i in range(1, max_iter):
            # Directly calculate residual every 'recalc' steps
            if (i % recalc) == 0:
                r = self.b - np.dot(self.A, x)
            else:
                # Else, update using one less matrix-vector product
                r -= a * Ar

            # Check if close enough
            if la.norm(r) <= tol: break

            # If not, take another step
            Ar = np.dot(self.A, r)
            x += a * r

        return x

    def path(self, tol = 10**-5, x_0 = None, max_iter=500, **kwargs):
        """
        Returns list of points traversed during descent.
        """
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])
        # ======================================================================
        if self.A is None or self.b is None:
            raise AttributeError('A and/or b haven\'t been set yet.')

        assert len(self.A) == len(self.A.T) == len(self.b)
        if x_0 is None:
            x = np.random.randn(len(self.A))
        else:
            x = np.copy(x_0)
        # ======================================================================
        path = [np.copy(x)]

        # First descent step ===================================================
        r = self.b - np.dot(self.A, x)
        # Check if close enough already
        if la.norm(r) <= tol:
            return x

        # If not, take a step
        Ar = np.dot(self.A, r)
        a = np.inner(r.T, r) / np.dot(r.T, Ar)
        x += a * r
        path.append(np.copy(x))

        # Rest of descent ======================================================
        for i in range(1, max_iter):
            # Directly calculate residual every 'recalc' steps
            if (i % recalc) == 0:
                r = self.b - np.dot(self.A, x)
            else:
                # Else, update using one less matrix-vector product
                r -= a * Ar

            # Check if close enough
            if la.norm(r) <= tol: break

            # If not, take another step
            Ar = np.dot(self.A, r)
            x += a * r
            path.append(np.copy(x))

        return path

class ConjugateGradientsSolver(Solver):
    """
    Conjugate gradients solver.

    Extra parameter(s) for Solver.solve(...):
        (int) recalc:   Directly recalculate the residual b - Ax every 'recalc'
                            iterations.
    """

    def __str__(self):
        l1 = 'Conjugate Gradients Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def __repr__(self):
        return self.__str__()


    def _full(self, tol, x, max_iter, x_true, **kwargs):
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])

        start_time = time.time()
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]

        # First descent step (gradient descent step) ===========================
        r = self.b - np.dot(self.A, x)
        r_norm = la.norm(r)
        residuals = [(r_norm, time.time() - start_time)]

        # Check if close enough already
        if r_norm <= tol:
            if x_true is None:
                return x, 0, residuals
            else:
                return x, 0, residuals, x_difs

        # If not, take a step
        rTr = np.dot(r.T, r)
        i = 1
        d = np.copy(r)  # First search direction is just the residual
        Ad = np.dot(self.A, d)
        a = rTr / np.dot(d.T, Ad)
        x += a * d
        if x_true is not None:
            x_difs.append(la.norm(x - x_true))
        # ======================================================================

        while i < max_iter:
            if (i % recalc) == 0:
                new_r = self.b - np.dot(self.A, x)
            else:
                new_r = r - (a * Ad)

            r_norm = la.norm(new_r)
            residuals.append((r_norm, time.time() - start_time))

            # Check if close enough
            if r_norm < tol:
                break

            i += 1

            # If not, take a step
            new_rTr = np.dot(new_r.T, new_r)
            beta = new_rTr / rTr

            d = new_r + beta * d
            r, rTr = new_r, new_rTr
            Ad = np.dot(self.A, d)

            a = rTr / np.dot(d.T, Ad)

            x += a * d
            if x_true is not None:
                x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals
        else:
            return x, i, residuals, x_difs

    def _bare(self, tol, x, max_iter, **kwargs):
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])

        # First descent step (GD) ==============================================
        r = self.b - np.dot(self.A, x)
        # Check if close enough already
        if la.norm(r) <= tol:
            return x

        # If not, take a step
        rTr = np.dot(r.T, r)
        d = np.copy(r)
        Ad = np.dot(self.A, d)
        a = rTr / np.dot(d.T, Ad)
        x += a * d

        for i in range(1, max_iter):
            if (i % recalc) == 0:
                new_r = self.b - np.dot(self.A, x)
            else:
                new_r = r - (a * Ad)

            if la.norm(new_r) <= tol:
                break

            new_rTr = np.dot(new_r.T, new_r)
            beta = new_rTr / rTr

            d = new_r + beta * d
            r, rTr = new_r, new_rTr
            Ad = np.dot(self.A, d)

            a = rTr / np.dot(d.T, Ad)

            x += a * d

        return x

    def path(self, tol = 10**-5, x_0=None, max_iter = 500, **kwargs):
        if 'recalc' not in kwargs:
            recalc = 20
        else:
            recalc = int(kwargs['recalc'])

        self._check_ready()
        if x_0 is None:
            x = np.zeros(len(self.A))
        else:
            x = np.copy(x_0)
        # ======================================================================
        path = [np.copy(x)]

        # First descent step (GD) ==============================================
        r = self.b - np.dot(self.A, x)
        # Check if close enough already
        if la.norm(r) <= tol:
            return path

        # If not, take a step
        rTr = np.dot(r.T, r)
        d = np.copy(r)
        Ad = np.dot(self.A, d)
        a = rTr / np.dot(d.T, Ad)
        x += a * d
        path.append(np.copy(x))

        for i in range(1, max_iter):
            if (i % recalc) == 0:
                new_r = self.b - np.dot(self.A, x)
            else:
                new_r = r - (a * Ad)

            if la.norm(new_r) <= tol:
                break

            new_rTr = np.dot(new_r.T, new_r)
            beta = new_rTr / rTr

            d = new_r + beta * d
            r, rTr = new_r, new_rTr
            Ad = np.dot(self.A, d)

            a = rTr / np.dot(d.T, Ad)

            x += a * d
            path.append(np.copy(x))

        return path

# TODO: add parameter governing epsilon's decay rate
class IterativeRefinementSolver(Solver):

    def __str__(self):
        l1 = 'Iterative Refinement Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def __repr__(self):
        return self.__str__()

    def _full(self, tol, x, max_iter, x_true, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])

        start_time = time.time()
        residuals = []
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]


        i = 0
        while i < max_iter:

            r = self.b - np.dot(self.A, x)
            r_norm = la.norm(r)
            residuals.append((r_norm, time.time() - start_time))

            if r_norm <= tol:
                break
            i += 1

            eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))


            x += np.dot(la.inv(A_e), r)

            if x_true is not None:
                x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals
        else:
            return x, i, residuals, x_difs

    def _bare(self, tol, x, max_iter, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])

        for i in range(max_iter):
            r = self.b - np.dot(self.A, x)

            if la.norm(r) <= tol:
                break

            eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))

            x += np.dot(la.inv(A_e), r)

        return x

    def path(self, tol=10**-5, x_0=None, max_iter = 500, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])



        self._check_ready()
        if x_0 is None:
            x = np.zeros(len(self.A))
        else:
            x = np.copy(x_0)

        path = [np.copy(x)]

        for i in range(max_iter):
            r = self.b - np.dot(self.A, x)

            if la.norm(r) <= tol:
                break

            eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))

            x += np.dot(la.inv(A_e), r)
            path.append(np.copy(x))


        return path

class IterativeRefinementGeneralSolver(Solver):

    def __str__(self):
        l1 = 'Iterative Refinement General Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (len(self.A), len(self.A.T))
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True\n'
        else:
            l3 = 'full_output: False'
        if self.intermediate_solver is None:
            l4 = 'intermediate_solver: None\n'
        else:
            l4 = 'intermediate_solver: %s' % self.intermediate_solver
        return l1+l2+l3+l4

    def __repr__(self):
        return self.__str__()

    def _full(self, tol, x, max_iter, x_true, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])

        start_time = time.time()
        residuals = []
        if x_true is not None:
            x_difs = [la.norm(x - x_true)]

        i = 0
        while i < max_iter:

            r = self.b - np.dot(self.A, x)
            r_norm = la.norm(r)
            residuals.append((r_norm, time.time() - start_time))

            if r_norm <= tol:
                break
            i += 1

            if self.intermediate_continuation == True:
                eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))

            ## call intermediate solver method
            solver_object = self.intermediate_solver(A_e,r,full_output=self.full_output)
            d_i, i_i, r_i, x_d_i = solver_object.solve(tol=10**-5, x_0=r, max_iter=self.intermediate_iter, recalc=20, x_true=x_true)
            # residuals.append((la.norm(r_i),time.time() - start_time))
            # x_difs.append(la.norm(x_d_i))

            ## update x
            x += d_i

            ## track
            if x_true is not None:
                x_difs.append(la.norm(x - x_true))

        if x_true is None:
            return x, i, residuals
        else:
            return x, i, residuals, x_difs

    def _bare(self, tol, x, max_iter, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])

        for i in range(max_iter):
            r = self.b - np.dot(self.A, x)

            if la.norm(r) <= tol:
                break

            if self.intermediate_continuation == True:
                eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))

            ## call intermediate solver method
            solver_object = self.intermediate_solver(A_e,r,full_output=self.full_output)
            d_i, i, r_i, x_d_i = solver_object.solve(tol=10**-5, x_0=r, max_iter=self.intermediate_iter, recalc=20, x_true=None)
            x += d_i

        return x

    def path(self, tol=10**-5, x_0=None, max_iter = 500, **kwargs):
        if 'eps' not in kwargs:
            eps = 2 * la.norm(self.A)
        else:
            eps = float(kwargs['eps'])

        self._check_ready()
        if x_0 is None:
            x = np.zeros(len(self.A))
        else:
            x = np.copy(x_0)

        path = [x]

        for i in range(max_iter):
            r = self.b - np.dot(self.A, x)

            if la.norm(r) <= tol:
                break

            if self.intermediate_continuation == True:
                eps *= 0.5
            A_e = self.A + eps * np.identity(len(self.A))

            ## call intermediate solver method
            solver_object = self.intermediate_solver(A_e,r,full_output=self.full_output)
            d_i, i, r_i = solver_object.solve(tol=10**-5, x_0=r, max_iter=self.intermediate_iter, recalc=20,x_true=None)
            x += d_i

            path.append(x)

        return x


# TO DO: BiCGStab

# This works, but much slower than CG for large/high condition number matrices
def iter_refinement_eps(A, b, tol=0.001, numIter=500, x=None, e=None, full_output=False):
    """
    Iterative refinement with epsilon smoothing.

    e: epsilon, value added to diagonal of A to lower condition number (decreases
                    w/ each iteration)
    """
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    n = len(ATA)
    if x is None:
        x = np.zeros(n)
    if e is None:
        e = 2*la.norm(ATA)

    #min_err = (np.copy(x), norm_dif(x, A, b))

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        e = 0.5 * e
        if np.random.uniform() < 0.01:
            print('IR iter %d; e: %f' % (i, e))


        r = ATb - np.dot(ATA, x)
        r_norm = la.norm(r)

        # break if residual blows up (becomes nan)
        if r_norm != r_norm:
            break

        if full_output:
            resids[time.time() - start_time] = r_norm

        # exit if close enough
        if r_norm < tol:
            if full_output:
                return x, i, True, resids
            else:
                return x

        A_e = np.copy(ATA) + e * np.identity(n)

        #print('ITER REFINE %d' % i)
        #d = gradient_descent_helper(np.copy(A_e), np.copy(r), np.copy(x))
        #d = conjugate_gradient(np.copy(A_e), np.copy(r), x=np.copy(x))


        x += np.dot(la.inv(A_e), r)

        #if norm_dif(x, A, b) < min_err[1]: min_err = (np.copy(x), norm_dif(x, A, b))


    #print('IR: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x




























# ==============================================================================

def conjugate_gs_alt(U, A):
    """
    Conjugate Gram-Schmidt process.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    Args:
        (numpy.ndarray) U: array of n linearly independent column vectors.
        (numpy.ndarray) A: matrix for vectors to be mutually conjugate to.

    Returns:
        (numpy.ndarray) D: array of n mutually A-conjugate column vectors.
    """
    n = len(U)
    D = np.copy(U)
    beta = np.zeros([n,n])

    D[:, 0] = U[:, 0]
    for i in range(1, n):
        for j in range(0,i-1):

            Adj = np.dot(A, D[:, j])

            beta[i, j] = -np.dot(U[:, i].T, Adj)
            beta[i, j] /= np.dot(D[:, j].T, Adj) # (37)

            D[:, i] = U[:, i] + np.dot(beta[i, j], D[:, j]) # (36)

    ## checks
    for i in range(0, n-1):
        for j in range(i+1,n):
            # print( np.dot(U[:, i],np.dot(A,D[:, j])) + beta[i, j]*np.dot(D[:, j].T,np.dot(A,D[:, j])) )
            print( np.dot(D[:,i], np.dot(A, D[:,j])) )

    return D

def conjugate_gs(u, A):
    """
    Conjugate Gram-Schmidt process.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    Args:
        (numpy.ndarray) u: array of n linearly independent column vectors.
        (numpy.ndarray) A: matrix for vectors to be mutually conjugate to.

    Returns:
        (numpy.ndarray) d: array of n mutually A-conjugate column vectors.
    """
    n = len(u)
    d = np.copy(u)

    for i in range(1, n):
        for j in range(0,i):

            Adj = np.dot(A, d[:, j])


            Bij = -np.inner(u[:, i].T, Adj)
            Bij /= np.inner(d[:, j].T, Adj) # (37)

            d[:, i] += np.dot(Bij, d[:, j]) # (36)

    return d

def arnoldi(A,b):
    """
    Conjugate (modifed) Gram-Schmidt process for Krylov(A,b).

    Args:
        A:  matrix (psd)
        b:  vector RHS solution

    Returns:
        Q:  matrix (unitary/orthogonal) normalized vectors where
            Q[:,1], ... , Q[:,n] span K^n(A,b)
        H:  matrix (upper hessenberg) s.t.
            H = Q^T A Q upon completion
    """

    ## initialize
    n = len(A)
    Q = np.zeros([n,n])
    H = np.zeros([n,n])

    ## first vector
    Q[:,0] = b/la.norm(b)

    ## remaining vectors
    for j in range(n-1):               # start computation for Q[:,j+1]
        t = np.dot(A,Q[:,j])           # t \in K^[j+1](A,b)
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i],t)  # H[i,j] * Q[:,i] is proj of t onto Q[:,i]
            t -= H[i,j] * Q[:,i]       # remove proj (ORTHO)
        H[j+1,j] = la.norm(t)
        Q[:,j+1] = t/H[j+1,j]          # normalize (NORMALIZE)

    ## last column of H
    H[:,n-1] = np.dot(Q.T,np.dot(A,Q[:,n-1]))
    return Q,H

def test_arnoldi(A,b):
    Q,H = arnoldi(A,b)
    for i in range(len(Q)):
        for j in range(i):
            print(np.dot(Q[:,i],Q[:,j]))
    print(H - np.dot(Q.T,np.dot(A,Q)))

def jacobi(A,b,tol=0.001,maxiter=1000,x_0=None):
    '''
    Solves Ax = b with Jacobi splitting method
        A \in R^[n x n]
        b,x \in R^n

    ONLY WORKS for matrices A such that spectral_radius(B) < 1, where
        B = D-1 E,
        D = diagonal elements of A (zero elsewhere),
        E = non-diagonal elements of A (zero on diagonal)

    '''

    n = A.shape[0]

    ## start
    if x_0 == None:
        x_0 = np.random.randn(n)

    ## construct matrix components
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i][i] = A[i][i]
    E = A-D
    Dinv = la.inv(D)
    B = np.dot(-Dinv,E)
    z = np.dot(Dinv,b)

    spec_rad = max(la.svd(B)[1])**2
    if spec_rad >= 1:
        print('Spectral radius of B (%f) >= 1. Method won\'t converge.' % spec_rad)
        print('Returning None.')
        return None
    else:
        print('Spectral radius of B: %f' % spec_rad)

    ## iterations
    x = x_0
    for i in range(maxiter):
        x = np.dot(B,x) + z
        #print(la.norm(np.dot(A,x)-b))
        if la.norm(np.dot(A,x)-b) <= tol:
            break

    return x

def iter_refinement(A, b, tol=0.001, numIter=500, x=None, full_output=False):
    """
    Iterative refinement method.

    https://en.wikipedia.org/wiki/Iterative_refinement

    Works, but needs more testing on various sizes, condition numbers + initial
    error in Ax=b.
    """
    # tol *= la.norm(A)

    m = len(A)
    n = len(A.T)
    if x is None:
        x = np.zeros(n)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        #print('Iter %d' % i)

        # Compute the residual r
        r = b - np.dot(A, x)

        if full_output:
            resids[time.time() - start_time] = la.norm(r)

        # Solve the system (Ad = r) for d
        result = scopt.minimize(fun=norm_dif, x_0=np.random.randn(m), \
                                args=(A, r), method='CG')
        d, success, msg = result.x, result.success, result.message
        # TODO: find out which method is best/quickest to solve this

        x += d


        if la.norm(b - np.dot(A, x)) < tol:
            print('IR: Close enough at iter %d' % i)
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    print('IR: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

# this doesn't work ever
def iter_refinement_const_eps(A, b, tol=0.001, numIter=500, x=None, e=None, full_output=False):
    assert len(A) == len(A.T)
    n = len(A)

    if e is None:
        print('MAGIC NUMEBR')
        e = la.norm(A) / 2.0 # MAGIC NUMBER
    if x is None:
        x = np.zeros(n)
    if full_output:
        resids = OrderedDict()
        st_time = time.time()

    A_eps = A + e*np.identity(n)

    for i in range(numIter):
        r = np.dot(A, x) - b
        r_norm = la.norm(r)

        if full_output:
            resids[time.time() - st_time] = r_norm

        if r_norm < tol:
            if full_output:
                return x, i, True, resids
            else:
                return x

        #d = conjugate_gradient(np.copy(A_eps), np.copy(r), x=np.copy(x))
        x -= np.dot(la.inv(A_eps), r)

    if full_output:
        return x, numIter, False, resids
    else:
        return x

# ==============================================================================
# GRADIENT DESCENT GRAVEYARD (?)

def gradient_descent_helper(A, b, x, alpha=0.01, tol=0.1, verbose=0):
    """
    Helper method for iter_refinement_eps (NOPE). Standard gradient descent that also
        works on non-symmetric matrices.
    """
    n_iter = 0
    start_time = time.time()

    while 1:
        #if np.random.uniform() <= 0.00001:
        if 1:
            print('n_iter: %d' % n_iter)
            print(norm_dif(x, A, b))

        err = np.dot(A, x) - b
        # check for nan
        if la.norm(err) == float('Inf'):
            print('something went horribly wrong in gradient_descent_helper')
            sys.exit(0)

        # return if close enough
        if la.norm(err) < tol:
            break

        gradient = np.dot(A.T, err) / len(A)
        # also return if not close enough, but gradient still ~= 0
        # (in case of overconstrained linear systems, for example)
        if la.norm(gradient) < 0.000001:
            break

        # update
        x -= alpha * gradient

        n_iter += 1


    if verbose:
        print('n_iter: %d' % n_iter)
        print('time: %f' % (time.time() - start_time))

    return x

# baseline; for symmetric, positive-definite A
def gradient_descent(A, b, tol=10**-5, x = None, numIter = 500, full_output=False):
    """
    Standard gradient descent for SYMMETRIC, POSITIVE-DEFINITE matrices.

    Re-calculates residual EVERY iteration (so slow but a bit more accurate).

    Args:
        numpy.ndarray A:    n x n transformation matrix.
        numpy.ndarray b:    n x 1 "target values".
        numpy.ndarray x:    n x 1 initial guess (optional).
        int     numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ab - x||.
    """
    n = len(A)
    if x is None: x = np.zeros(n)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    # Start descent
    for i in range(numIter):
        if full_output:
            resids[time.time() - start_time] = norm_dif(x, A, b)

        # ACTUAL ALGORITHM
        # ======================================================================
        # calculate residual (direction of steepest descent)
        r = b - np.dot(A, x)

        # calculate step size (via line search)
        a = np.inner(r.T, r) / float(np.inner(r.T, np.inner(A, r)))

        # update x
        x += a * r
        # ======================================================================

        if la.norm(b - np.dot(A, x)) < tol:
            print('GD: Close enough at iter %d' % i)
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    print('GD: Max iteration reached (%d)' % numIter)
    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

# modification: 1 matrix-vector multiplication per iteration
def gradient_descent_alt(A, b, x_0=None, x_tru=None, tol=10**-5, numIter=500, recalc=50, full_output=False):
    """
    Implementation of gradient descent for PSD matrices.

    Notes:
        Needs thorough testing.
        Only 1 matrix-vector computation is performed per iteration (vs 2).
        Slow history tracking.

    Args:
        (numpy.ndarray)     A:    n x n transformation matrix.
        (numpy.ndarray)     b:    n x 1 "target values".
        (numpy.ndarray)    x_0:    n x 1 initial guess (optional).
        (numpy.ndarray) x_tru:    n x 1 true x (optional).
        (int)         numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ax - b||_2.
    """

    n = len(A)

    # Ensure sound inputs
    assert len(A.T) == n
    assert len(b) == n

    # Working with (n, ) vectors, not (n, 1)
    if len(b.shape) == 2: b = b.reshape(n, )
    if x_0 is None:
        x_0 = np.random.randn(n, )
    else:
        assert len(x_0) == n
        if len(x_0.shape) == 2: x_0 = x_0.reshape(n, ) # (n, ) over (n, 1)

    # diagnostics
    x_hist = []

    if full_output:
        resids = []

    # first descent step
    x = x_0
    r_curr = b - np.dot(A, x)
    Ar_curr = np.dot(A,r_curr)
    a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))
    r_new = r_curr - a*Ar_curr
    x += a * r_curr

    if full_output:
        x_hist.append(x)
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        resids.append(err)

    # remaining descent steps
    for _ in range(1,numIter):

        # calculate residual (direction of steepest descent)
        r_curr = r_new

        # calculate step size (via analytic line search)
        Ar_curr = np.inner(A, r_curr)
        a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))

        # updates
        x += a * r_curr
        x_hist.append(x)

        # calculate residuals for next step
        if _ % recalc == 0:
            r_new = b - np.dot(A, x)
        else:
            r_new = r_curr - a*Ar_curr

        # add residuals
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        if full_output:
            resids.append(err)

        # stop if close
        if err < tol:
            print('GD_alt: Close enough at iter %d' % _)
            print(la.norm(r_new))
            if full_output:
                return x, _, True, resids
            else:
                return x

    print('GD_alt: Max iteration reached (%d)' % numIter)
    if full_output:
        return x, numIter, False, resids
    else:
        return x

# modifications: 1 matrix-vector multiplication per iteration; nonsymmetric (square) matrix A
def gradient_descent_nonsymm(A, b, x_0=None, x_tru=None, tol=10**-5, numIter=500, recalc=50, full_output=False):
    """
    Implementation of gradient descent for nonsymmetric matrices (or symmetric, but slow in this case).

    Notes:
        Needs thorough testing; error BLOW UP
        Re-calculate residual EVERY iteration (so slow but a bit more accurate).
        Only 1 matrix-vector computation is performed per iteration (vs 2).
        Slow history tracking.

    Args:
        (numpy.ndarray)     A:    n x n transformation matrix.
        (numpy.ndarray)     b:    n x 1 "target values".
        (numpy.ndarray)    x_0:    n x 1 initial guess (optional).
        (numpy.ndarray) x_tru:    n x 1 true x (optional).
        (int)         numIter:    Number of passes over data.

    Returns:
        argmin(x) ||Ax - b||_2.
    """

    n = len(A)

    # Ensure sound inputs
    assert len(A.T) == n
    assert len(b) == n

    # Working with (n, ) vectors, not (n, 1)
    if len(b.shape) == 2: b = b.reshape(n, )
    if x_0 is None:
        x_0 = np.random.randn(n, )
    else:
        assert len(x_0) == n
        if len(x_0.shape) == 2: x_0 = x_0.reshape(n, ) # (n, ) over (n, 1)

    # diagnostics
    x_hist = []

    if full_output:
        resids = []

    # first descent step
    x = x_0
    AA = 1/2*A+A.T
    r_curr = b - np.dot(AA, x)
    Ar_curr = np.dot(AA,r_curr)
    a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))
    r_new = r_curr - a*Ar_curr
    x += a * r_curr

    if full_output:
        x_hist.append(x)
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        resids.append(err)

    # remaining descent steps
    for _ in range(1,numIter):

        # calculate residual (direction of steepest descent)
        r_curr = r_new

        # calculate step size (via analytic line search)
        AA = 1/2*A+A.T
        Ar_curr = np.inner(AA, r_curr)
        a = np.inner(r_curr.T, r_curr) / float(np.inner(r_curr.T, Ar_curr))

        # updates
        x += a * r_curr
        x_hist.append(x)

        # calculate residuals for next step
        if _ % recalc == 0:
            r_new = b - np.dot(AA, x)
        else:
            r_new = r_curr - a*Ar_curr

        # add residuals
        if x_tru is not None:
            err = la.norm(x-x_tru)
        else:
            err = la.norm(np.dot(A,x)-b)
        if full_output:
            resids.append(err)

        # stop if close
        if err < tol:
            print('GD_alt: Close enough at iter %d' % _)
            print(la.norm(r_new))
            if full_output:
                return x, _, True, resids
            else:
                return x

    print('GD_alt: Max iteration reached (%d)' % numIter)
    if full_output:
        return x, numIter, False, resids
    else:
        return x

# ==============================================================================
# CONJUGATE GRADIENT GRAVEYARD (?)
# for symmetric, positive-definite A
def conjugate_gradient_ideal(A, b, tol=0.001, x = None, numIter = 500, full_output=False):
    """
    For SYMMETRIC, POSITIVE-DEFINITE matrices.
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (p. 32)

    Tested on a handful of small (~50x50 - 500x500 matrices) w/ various
    condition numbers. Behaviour is as expected - systems with higher
    condition numbers take longer to solve accurately.

    TODO: fix residual error accumulation

    Returns:
        If not full_output: just the optimal x.
        If full_output: optimal x, num iterations taken, success, residuals plot.
    """
    #tol *= la.norm(A)

    m, n = len(A), len(A.T)

    if x is None:
        x = np.zeros(n)

    # d: first search direction (same as initial residual)
    d = b - np.dot(A, x) # d(0) = r(0) = b - Ax(0)
    r = d                # from eq. (45)

    if full_output:
        resids = OrderedDict()
        start_time = time.time()

    for i in range(numIter):
        if full_output:
            resids[time.time() - start_time] = norm_dif(x, A, b)

        # TODO: recalculate residual here every _ iters to avoid accumulating error
        # if 0:
        #     print(('r(%d): ' + str(r)) % i)
        #     recalc_r = b - np.dot(A, x)
        #     print('recalc: ' + str(recalc_r))
        #     print('resid dif: %f' % la.norm(r - recalc_r))


        a = np.dot(r.T, r) / np.dot(d.T, np.dot(A, d)) # eq. (46)

        x += a * d

        new_r = r - (a * np.dot(A, d)) # calculate new residual (A-orthogonal to
                                       # previous except d)      (eq. 47)

        beta = np.dot(new_r.T, new_r) / np.dot(r.T, r) # eq. (48)

        d = new_r + beta * d
        r = new_r

        if la.norm(b - np.dot(A, x)) < tol:
            if full_output:
                resids[time.time() - start_time] = norm_dif(x, A, b)
                return x, i, True, resids
            else:
                return x

    if full_output:
        resids[time.time() - start_time] = norm_dif(x, A, b)
        return x, numIter, False, resids
    else:
        return x

def conjugate_gradient_psd(A,b,x_0=None,x_tru=None,tol=10**-3,max_iter=500,recalc=50,full_output=False):
    """
    CG for symmetric, psd A with 1 matrix-vector multiplication per iteration
    """
    n = len(A)
    if x_0 is None:
        x_0 = np.random.randn(n)

    x = np.copy(x_0)
    i = 0
    r = b-np.dot(A,x)
    d = np.copy(r)
    del_new = np.dot(r,r)
    del_0 = np.copy(del_new)

    if full_output == True:
        resids = OrderedDict()
        start_time = time.time()
        resids[i] = la.norm(b-np.dot(A,x))
        if x_tru is not None:
            errs = OrderedDict()
            errs[i] = la.norm(x-x_tru)

    while not (i > max_iter or del_new < (tol**2)*del_0):

        q = np.dot(A,d)
        alpha = del_new / np.dot(d,q)
        x += alpha*d
        if i % recalc == 0:
            r = b-np.dot(A,x)
        else:
            r -= alpha*q

        ## updates
        del_old = np.copy(del_new)
        del_new = np.dot(r,r)
        beta = del_new / del_old
        d = r + beta*d
        i += 1

        if full_output == True:
            resids[i] = la.norm(b-np.dot(A,x))
            if x_tru is not None:
                errs[i] = la.norm(x-x_tru)

    if full_output == True:
        resids[i] = la.norm(b-np.dot(A,x))
        if i < max_iter:
            status = True
        else:
            status = False
        if x_tru is not None:
            errs[i] = la.norm(x-x_tru)
            return x, i, status, resids, errs
        else:
            return x, i, status, resids
    else:
        return x

# for any A
def conjugate_gradient(A, b, tol=0.001, x = None, numIter = 500, full_output=False):
    """
    Conjugate gradients on the normal equations.
    (Page 41 in "Painless Conjugate Gradient")

    A doesn't need to be symmetric, positive-definite, or even square.
    Use conjugate_gradient_ideal for matrices that satisfy the above conditions.
    """
    return conjugate_gradient_ideal(A = np.dot(A.T, A), \
                                    b = np.dot(A.T, b), x = x, \
                                    numIter = numIter, full_output=full_output)
