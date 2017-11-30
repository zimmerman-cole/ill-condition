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
        print("    f(x_nt) = {:s}".format(util.f(xs[-1])))
        print("    x_cp = {:s}".format(xs_cp[-1]))
        print("    f(x_cp) = {:0.2f}".format(util.f(xs_cp[-1])))
    print("===== upon completion ==========================")
    print("================================================")

    if show:
        util.visualize(xs, ss, lams, ms, xs_cp, mu0)
        return xs, ss, lams, ms, xs_cp, mu0
    return xs, ss, lams, ms

def simulate(x0=None, mu0=None, lam0=None, s0=None, K=10):
    N = len(x0)
    if len(mu0) < N:
        mu0 = np.repeat(mu0, N)
    if len(lam0) < N:
        lam0 = np.repeat(lam0, N)
    if len(mu0) < N:
        s0 = np.repeat(s0, N)
    xss, sss, lamss, mss, xss_cp, mu0s = [], [], [], [], [], []
    for i in range(len(x0)):
        xs, ss, lams, ms, xs_cp, mu00 = ipm(x=x0[i], s=s0[i], lam=lam0[i], mu=mu0[i], K=K)
        xss.append(xs)
        sss.append(ss)
        lamss.append(lams)
        mss.append(ms)
        xss_cp.append(xs_cp)
        mu0s.append(mu00)
    util.visualize_sims(xss, sss, lamss, mss, xss_cp, mu0s)


if __name__ == '__main__':
    x = [-1., 10.]
    s = 2.4375
    lam = 2
    mu = 50
    ipm(x=x, s=s, lam=lam, mu=mu, K=15)

    # x0 = [[-2.,20.], [-1.,20.], [-0.,20.], [1.,20.], [2.,20.]]
    # s0 = [2.4375, 2.4375, 2.4375, 2.4375, 2.4375]
    # lam0 = [2., 2., 2., 2., 2.]
    # mu0 = [100, 100, 100, 100, 100]
    # simulate(x0=x0, mu0=mu0, lam0=lam0, s0=s0)
