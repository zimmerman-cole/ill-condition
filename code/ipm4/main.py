import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util
## =============================================================================
## reference: http://apmonitor.com/me575/index.php/Main/BookChapters
## =============================================================================

def ipm(x=None, s=None, lam=None, mu=None, K=8, d=5., show=True):
    ## initialize
    x, s, lam, mu = util.parse_input(x, s, lam, mu)
    mu0 = mu
    xs, ss, lams, ms = [], [], [], []
    xs_cp = []
    F = util.F(x, s, lam, mu)
    if show:
        xs.append(np.copy(x))
        ss.append(np.copy(s))
        lams.append(np.copy(lam))
        ms.append(la.norm(F,2))
        xs_cp.append(np.copy(x))

    print("================================================")
    print("===== at initialization ========================")
    util.fun_status(x, s, lam, mu)
    print("===== at initialization ========================")
    print("================================================\n")

    for k in range(K):
        print("===== start iter {:d} =============================".format(k))
        util.status(x, s, lam, mu)

        ## solve
        F = util.F(x, s, lam, mu)
        J_F = util.J_F(x, s, lam, mu)
        p = la.solve(J_F, -F)

        ## update with full step for tracking
        if show:
            x_cp = util.get_cp(np.copy(x), np.copy(mu))
            xs_cp.append(x_cp)

        ## update and step
        x, s, lam = util.step(x, s, lam, p)

        if show:
            xs.append(np.copy(x))
            ss.append(np.copy(s))
            lams.append(np.copy(lam))
            ms.append(la.norm(F,2))
        mu /= d
        print("===== end iter {:d} ===============================\n".format(k))

    print("================================================")
    print("===== upon completion ==========================")
    util.fun_status(x, s, lam, mu)
    print("------------------------------------------------")
    if show:
        print("    x_nt = {:s}".format(xs[-1].T))
        print("    x_cp = {:s}".format(xs_cp[-1]))
    print("===== upon completion ==========================")
    print("================================================")

    if show:
        util.visualize(xs, ss, lams, ms, xs_cp, mu0)

if __name__ == '__main__':
    x = [-1., 10.]
    s = 2.4375
    lam = 2
    mu = 50
    ipm(x=x, s=s, lam=lam, mu=mu, K=10)
