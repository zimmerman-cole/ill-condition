import numpy as np
import numpy.linalg as la
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

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

def bfgs(A, b, H=None, tol=1.0, max_iter=500):
    """
    Page 140/Algorithm 6.1 in Nocedal and Wright.
    Also see the Implementation section on pages 142-143.
    """
    # =======================================================
    n = A.shape[0]          # A is symmetric (n x n)
    if sps.issparse(A):
        iden = sps.eye
    else:
        iden = la.identity

    if H is None:
        H = iden(n)

    # =======================================================

    k = 0
    x = np.zeros(n)

    gr = A.dot(x) - b           # gradient
    #gr_norm = la.norm(gr)
    residuals = [la.norm(gr)]

    while la.norm(gr) > tol:
        print('Iter %d' % k)

        p = -H.dot(gr)          # search direction (6.18)

        # ===================================================
        # TODO: FIND BEST WAY TO DETERMINE STEP SIZE THAT
        # TODO: SATISFIES WOLFE CONDITIONS.
        # Find step size:
        a = 1.0
        x_new = x + (a * p)
        gr_new = A.dot(x) - b

        # Check Wolfe   (c1,c2 from Nocedal+Wright)
        n_tries = 1
        while not wolfe(a=a, c1=10**-4, c2=0.9, A=A, b=b, x=x, \
                            x_new=x_new, p=p, gr=gr, gr_new=gr_new):
            print('Wolfe try %d' % n_tries)
            n_tries += 1

            a *= 2
            x_new = x + (a * p)
            gr_new = A.dot(x) - b

        residuals.append(la.norm(gr_new))
        # ===================================================

        s = x_new - x
        y = gr_new - gr
        # ===================================================

        # COMPUTE Hk+1 BY MEANS OF (6.17)
        rho = 1.0 / np.inner(y.T, s)    # <== (6.14)

        H = ( iden(n) -rho*np.outer(s, y.T) ).dot( H.dot( iden(n) - rho*np.outer(y, s.T) ) )
        H += rho * np.outer(s, s.T)     # <== (6.17) ^^

        # ===================================================
        k += 1
        x, gr = x_new, gr_new

    return x, k, residuals

if __name__ == "__main__":
    X = sps.random(m=100, n=100, density=0.02)
    f_true = np.array([50 if 40<=i and i<60 else 0 for i in range(100)])
    g = X.dot(f_true)

    print('Init resid err: %f' % la.norm(g - X.dot(np.zeros(100))))
    fopt, n_iter, residuals = bfgs(A=X, b=g)
    print('Final resid err: %f' % la.norm(g - X.dot(fopt)))
    print('Took %d iter' % n_iter)

    plt.plot(residuals)
    plt.yscale('log')
    plt.show()
