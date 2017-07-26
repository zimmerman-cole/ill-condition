import numpy as np
import numpy.linalg as la
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
import optimize, time
import sys, util
from tomo2D import drt as drt



def f_eval(A,b,x):
    """
    Objective function f(x) = (1/2)(x.T A x) - (b.T x)
    """
    f = 0.5*x.dot(A.dot(x)) - b.dot(x)
    return f

def btls(A, b, x, p, g, alpha=1, rho=0.1, c=0.9):
    """
    Backtracking line-search. Takes small/conservative
        steps.
    Does NOT guarantee the sufficient decrease condition
        (3.6a from Nocedal+Wright) is satisfied.

    Args:
        alpha: initial trial step size
        rho: decrement factor
        c: additional decrease factor
    """
    f_curr = f_eval(A,b,x)
    diff = 1.0
    i = 0
    while diff > 0:
        x_proposed = x + alpha*np.copy(p)
        x_proposed = x_proposed.reshape(len(x), )
        f_proposed = f_eval(A=A,b=b,x=x_proposed)
        f_compare = np.copy(f_curr) + c*alpha*g.dot(p)
        diff = f_proposed - f_compare
        i += 1
        alpha *= rho
        if alpha < 10**-16:
            print("alpha: machine precision reached")
            break
    return alpha

def wolfe(a, c1, c2, A, b, x, x_new, p, gr, gr_new):
    """
    Determines whether step size 'a' satisfies Wolfe conditions (not strong).

    Args:
             a:     Step size.
        c1, c2:     Constants from (0,1).
          A, b:     From Ax=b.
             x:     Current estimate of optimal x.
         x_new:     Next (prospective) x.
             p:     Current search direction.
            gr:     Current gradient (Ax - b).
        gr_new:     Gradient at (prospective) next x.
    """
    # condition 1
    lhs = la.norm( b - A.dot(x_new) )
    rhs = la.norm(b - A.dot(x)) + c1*a*np.inner(p.T, gr)

    if lhs > rhs: return False

    # condition 2
    lhs = -np.inner(p.T, gr_new)
    rhs = -c2*np.inner(p.T, gr)

    return lhs <= rhs

def bfgs(A, b, H=None, B=1.0, tol=10**-5, max_iter=500, x_true=None):
    """
    Page 140/Algorithm 6.1 in Nocedal and Wright.
    Also see the Implementation section on pages 142-143.

    Doesn't do to well on large systems:
        8+ matrix-vector ops per iteration, many of which involve
        the very dense inverse Hessian approximation H.
    """
    # =======================================================
    n = A.shape[0]          # A is symmetric (n x n)
    if sps.issparse(A):
        iden = sps.eye
    else:
        iden = np.identity

    # Initialize H as BI
    if H is None:
        H = B * iden(n)     # inverse Hessian approximation H0

    # =======================================================

    k = 0
    x = np.zeros(n)
    exes = [x]
    if x_true is not None:
        x_difs = [la.norm(x_true - x)]

    start_time = time.time()

    gr = A.dot(x) - b           # gradient
    gr_norm = la.norm(gr)
    residuals = [(gr_norm, time.time() - start_time)]   # residual = -gradient
    # OPTIMIZED
    p = np.array(-H.dot(gr)).reshape(n, )   # (6.18) search direction

    while gr_norm > tol:

        # ===================================================
        # TODO: Best way to det. step size
        # TODO: DAMPED BFGS (for when curvature doesn't change much)
        Ap = A.dot(p)
        a = (np.inner(b, p) - np.inner(x, Ap)) / (np.inner(p, Ap))
        #print('step size at %d: %f' % (k, a))

        # ===================================================
        # Then update x, calculate new gradient
        x_new = x + a*p
        gr_new = A.dot(x_new) - b
        gr_norm = la.norm(gr_new)

        residuals.append((gr_norm, time.time() - start_time))
        if x_true is not None:
            x_difs.append(la.norm(x_true - x_new))

        # ===================================================
        # Calculate x-step, gradient-change
        s = x_new - x
        y = gr_new - gr
        # ===================================================
        # Update your inverse Hessian approximation
        # COMPUTE Hk+1 BY MEANS OF (6.17)
        rho = 1.0 / np.inner(y.T, s)    # <== (6.14)

        # OPTIMIZED
        Hy = H.dot(y)
        #print(Hy.shape, type(Hy))
        Hy = np.array(Hy).reshape(n,)
        yHy = y.dot(Hy)
        HysT = np.outer(Hy,s)
        rssT = rho*np.outer(s,s)

        H = H - rho*HysT - rho*HysT.T + rho*yHy*rssT + rssT

        # ===================================================
        # Then calculate your new search direction
        p = -H.dot(gr_new)
        p = np.array(p).reshape(n,)


        # ===================================================
        k += 1
        x, gr = x_new, gr_new
        exes.append(x)
        if k >= max_iter:
            break

    if x_true is None:
        return x, k, residuals, exes
    else:
        return x, k, residuals, exes, x_difs

# Test BFGS on "realistic" system matrix
def bfgs_system(n=100, m=100):
    """
    Test BFGS (vs. GD and CG) on "realistic" sparse system matrix.

    (int)   m:  number of rays to fire
    (int)   n:  image is (n x n) pixels

    Solves the system Xf=g, where
        X:  (m   , n**2)
        f:  (n**2,     )
        g:  (m   ,     )
    """
    X = drt.gen_X(n_1=n, n_2=n, m=m, sp_rep=1)
    X = sps.csr_matrix(X.T.dot(X))

    f_true = np.array([100 if (0.4*(n**2))<=i and i<(0.6*(n**2)) else 0 for i in range(n**2)])
    g = X.dot(f_true)

    print('Init resid err: %f' % la.norm(g - X.dot(np.zeros(n**2))))
    fopt, n_iter, b_resids, exes = bfgs(A=X, b=g, B=2.0, max_iter=1000)
    print('BFGS final resid err: %f' % la.norm(g - X.dot(fopt)))
    print('BFGS took %d iter' % n_iter)

    cgs = optimize.ConjugateGradientsSolver(A=X, b=g, full_output=1)
    fopt, n_iter, cg_resids = cgs.solve(max_iter=1000)
    print('CG final resid err: %f' % la.norm(g - X.dot(fopt)))
    print('CG took %d iter' % n_iter)

    gds = optimize.GradientDescentSolver(A=X, b=g, full_output=1)
    fopt, n_iter, gd_resids = gds.solve(max_iter=1000)
    print('GD final resid err: %f' % la.norm(g - X.dot(fopt)))
    print('GD took %d iter' % n_iter)

    plt.plot([t for n,t in b_resids], [n for n,t in b_resids], marker='o')
    plt.plot([t for n,t in cg_resids], [n for n,t in cg_resids], marker='o')
    plt.plot([t for n,t in gd_resids], [n for n,t in gd_resids], marker='o')
    plt.title('RESIDUALS')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Residual norm')
    plt.legend(['BFGS', 'CG', 'GD'])
    plt.show()

def l_bfgs(A, b, m=10, tol=10**-5, max_iter=500, x_true=None):
    """
    Limited-memory BFGS (Nocedal and Wright pg. 177).
        Based off algorithms 7.5 (& 7.4) from N and W.

    Instead of explicitly tracking H, this algorithm tracks the last m
        set of vectors {si, yi} (tracking curvature information from last
        m iterations). At each iteration, discard oldest vector pair and
        replace it with new pair.

    The product (Hk * grad) can be computed efficiently using these
        vector pairs.
    """
    n = A.shape[0]
    x = np.zeros(n,)

    # Initialize m vector-pairs (all as zeros)
    ys = [np.zeros(n) for _ in range(m)]    # last m y-vectors
    ss = [np.zeros(n) for _ in range(m)]    # last m s-vectors
    rhos = [0 for _ in range(m)]            # corresponding rhos

    start_time = time.time()
    k = 0

    gr = A.dot(x) - b   # initial gradient
    gr_norm = la.norm(gr)

    residuals = [(gr_norm, time.time() - start_time)]

    # setup iteration (done outside main loop) =================================
    # (does a step of gradient descent then saves relevant data)
    p = -gr

    # calculate step size ======
    Ap = A.dot(p)
    a = (np.inner(b, p) - np.inner(x, Ap)) / (np.inner(p, Ap))
    #print('step size at setup: %f' % a)

    # update x =================
    x_new = x + a*p
    gr_new = A.dot(x_new) - b
    gr_norm = la.norm(gr_new)
    residuals.append((gr_norm, time.time() - start_time))

    ys[0] = gr_new - gr
    ss[0] = x_new - x
    rhos[0] = 1.0 / np.inner(ys[0], ss[0])

    ix = 1      # index of next (oldest) vector-pair to be replaced

    k += 1
    x, gr = x_new, gr_new
    # ==========================================================================

    def Hgrad(q, ys, ss, rhos):
        """
        Computes the product (Hk * gradient) at each iteration.
            Based off algorithm 7.4 from Nocedal & Wright.

          gr:     gradient of obj. function at current x
          ys:     last m y-vectors
          ss:     last m s-vectors
        rhos:     vector-pairs' corresponding rhos
        """
        # Calculate Hk0 using (7.20)
        gamma = np.inner(  ss[(k-1)%m],   ys[(k-1)%m]  )   /   (la.norm(ss[(k-1)%m])**2)
        H = (gamma * sps.eye(n)).tocsr()

        alphas = [0 for _ in range(m)]
        for i in range(min(k, m)):
            alphas[i] = rhos[i] * np.inner(ss[i], q)
            q -= alphas[i] * ys[i]

        r = H.dot(q)

        for i in range(min(k, m)):
            B = rhos[i] * np.inner(ys[i], r)
            r += ss[i] * (alphas[i]-B)

        return r


    while k < max_iter:
        if gr_norm < tol:
            break

        # Chooses Hk0, then computes the product (Hk * gr) =====
        p = -Hgrad(gr, ys, ss, rhos)

        # Calculate step size ==================================
        Ap = A.dot(p)
        a = (np.inner(b, p) - np.inner(x, Ap)) / (np.inner(p, Ap))
        #print('step size at %d: %f' % (k, a))

        # Update x =============================================
        x_new = x + a*p
        gr_new = A.dot(x_new) - b
        gr_norm = la.norm(gr_new)
        residuals.append((gr_norm, time.time() - start_time))

        # Replace oldest vector-pair with new information ======
        ys[ix] = gr_new - gr
        ss[ix] = x_new - x
        rhos[ix] = 1.0 / np.inner(ys[ix], ss[ix])   # (7.17)

        # increment index of next vector-pair to be replaced ===
        ix = (ix+1) % m

        k += 1
        x, gr = x_new, gr_new

    return x, k, residuals

def l_bfgs_resids(n=100, plot_bfgs=False, sp=1):
    """
    L-BFGS residuals plotted (vs. BFGS and CG residuals)
    """

    if sp:
        A = sps.random(n, n)
        A = A.T.dot(A)
    else:
        A = util.psd_from_cond(cond_num=10**8, n=1000)

    x_true = np.array([100 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    b = A.dot(x_true)


    xopt, n_iter, lb_resids = l_bfgs(A=A, b=b)
    print('L-BFGS took %d iter' % n_iter)

    if plot_bfgs:
        xopt, n_iter, b_resids, b_path = bfgs(A=A, b=b, B=2.0)
        print('BFGS took %d iter' % n_iter)

    cgs = optimize.ConjugateGradientsSolver(A=A, b=b, full_output=1)
    xopt, n_iter, cg_resids = cgs.solve()
    print('CG took %d iter' % n_iter)

    gds = optimize.GradientDescentSolver(A=A, b=b, full_output=1)
    xopt, n_iter, gd_resids = gds.solve()
    print('GD took %d iter' % n_iter)

    leg = ['L-BFGS', 'CG', 'GD']
    plt.plot([t for n,t in lb_resids], [n for n,t in lb_resids], marker='o')
    plt.plot([t for n,t in cg_resids], [n for n,t in cg_resids], marker='o')
    plt.plot([t for n,t in gd_resids], [n for n,t in gd_resids], marker='o')
    if plot_bfgs:
        plt.plot([t for n,t in b_resids], [n for n,t in b_resids], marker='o')
        leg.append('BFGS')

    plt.ylabel('Residual norm')
    plt.xlabel('Time')
    plt.legend(leg)
    plt.yscale('log')
    plt.show()

def l_bfgs_system(n=1000, m=100):
    """
    Test L-BFGS on "realistic" system matrix.

    (int)   n: image is (n x n) pixels.
    (int)   m: number of rays to fire.
    """
    X = drt.gen_X(n_1=n, n_2=n, m=m, sp_rep=1)
    X = X.T.dot(X)

    f_true = np.array([100 if (0.4*(n**2))<=i and i<(0.6*(n**2)) else 0 for i in range(n**2)])
    g = X.dot(f_true)

    xopt, n_iter, lb_resids = l_bfgs(A=X, b=g)
    print('L-BFGS took %d iter' % n_iter)

    cgs = optimize.ConjugateGradientsSolver(A=X, b=g, full_output=1)
    xopt, n_iter, cg_resids = cgs.solve()
    print('CG took %d iter' % n_iter)

    gds = optimize.GradientDescentSolver(A=X, b=g, full_output=1)
    xopt, n_iter, gd_resids = gds.solve()
    print('GD took %d iter' % n_iter)

    plt.plot([t for n,t in lb_resids], [n for n,t in lb_resids], marker='o')
    plt.plot([t for n,t in cg_resids], [n for n,t in cg_resids], marker='o')
    plt.plot([t for n,t in gd_resids], [n for n,t in gd_resids], marker='o')
    plt.legend(['L-BFGS', 'CG', 'GD'])
    plt.yscale('log')
    plt.ylabel('Residual Norm')
    plt.xlabel('Time (sec)')
    plt.show()

class BFGSSolver(optimize.Solver):
    """
    Unfinished
    """

    def __str__(self):
        l1 = 'BFGS Solver\n'
        if self.A is None:
            l2 = 'A: None; '
        else:
            l2 = 'A: %d x %d; ' % (self.A.shape[0], self.A.shape[1])
        if self.b is None:
            l2 += 'b: None\n'
        else:
            l2 += 'b: %d x %d\n' % (len(self.b), len(self.b.T))
        if self.full_output:
            l3 = 'full_output: True'
        else:
            l3 = 'full_output: False'
        return l1+l2+l3

    def _check_ready(self):
        """
        Ensure A is square, dimensions line up w/ b.

        """

        assert self.A.shape[0] == self.A.shape[1]
        assert b.shape == (A.shape[0], )


    def _full(self, tol, x, max_iter, x_true, **kwargs):
        n = self.A.shape[0]          # A is symmetric (n x n)
        if sps.issparse(self.A):
            iden = sps.eye
        else:
            iden = np.identity

        if 'H' not in kwargs:
            if 'B' not in kwargs:
                B = 1.0

            H = B * iden(n) # inverse Hessian approximation

        k = 0



if __name__ == "__main__":
    bfgs_system()
