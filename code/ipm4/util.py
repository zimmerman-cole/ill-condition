import numpy as np
import numpy.linalg as la
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, linewidth=80)

## ===== primary ===============================================================
def f(x):
    """
    obj function
    """
    x1, x2 = x[0], x[1]
    return x1**4 - 2.*(x2 * x1**2) + x2**2 + x1**2 - 2.*x1 + 5.

def fp(x1, x2):
    """
    obj function (for plotting)
    """
    return x1**4 - 2.*(x2 * x1**2) + x2**2 + x1**2 - 2.*x1 + 5.

def g(x):
    """
    constraints (m=1)
    """
    x1, x2 = x[0], x[1]
    return -(x1 + 0.25)**2 + 0.75*x2
## ===== primary ===============================================================

## ===== derivs ================================================================
def grad_f(x):
    """
    grad of obj function
    """
    x1, x2 = x[0], x[1]
    out = np.zeros([2, 1])
    out[0] = 4.*x1**3 - 4*(x1 * x2) + 2.*x1 - 2
    out[1] = -2.*x1**2 + 2.*x2
    return out

def grad_g(x):
    """
    grad of constraints (m=1)
    """
    x1, x2 = x[0], x[1]
    m = 1
    out = np.zeros([2, m])   # note: would be n-x-m for m constraints
    out[0] = -2.*(x1 + 0.25)
    out[1] = 0.75
    return out
## ===== derivs ================================================================

## ===== second ================================================================
def hess_f(x):
    """
    hess of obj function
    """
    x1, x2 = x[0], x[1]
    out = np.zeros([2, 2])
    out[0,0] = 12.*x1**2 - 4.*x2 + 2
    out[0,1] = -4.*x1
    out[1,0] = -4.*x1
    out[1,1] = 2
    return out

def hess_g(x):
    """
    hess of constraints (m=1)
    """
    x1, x2 = x[0], x[1]
    out = np.zeros([2, 2])
    out[0,0] = -2.
    out[0,1] = 0.
    out[1,0] = 0.
    out[1,1] = 0.
    return out

## ===== other =================================================================
def dmat(v):
    """
    place vector on diagonal
    """
    return np.diag(v)

def parse_input(x, s, lam, mu, m=1, n=2):
    m_param = [s, lam]
    xx = np.zeros([n, 1])
    xx[0], xx[1] = x[0], x[1]
    s, lam = [np.array([float(p)]) for p in m_param]
    mu = float(mu)
    return xx, s, lam, mu

def fun_status(x, s, lam, mu):
    print("----- primary -----------------------------")
    print("    f = {:s}".format(f(x)))
    print("    g = {:s}".format(g(x)))
    print("----- primary -----------------------------\n")
    print("----- deriv -------------------------------")
    print("    grad_f = \n{:s}".format(grad_f(x)))
    print("    grad_g = \n{:s}".format(grad_g(x)))
    print("----- deriv -------------------------------\n")
    print("----- rootfinding -------------------------")
    print("    F = \n{:s}".format(F(x, s, lam, mu)))
    print("    JF = \n{:s}".format(J_F(x, s, lam, mu)))
    print("----- rootfinding -------------------------")

def status(x, s, lam, mu):
    print("----- point -------------------------------")
    print("    x = {:s}".format(x.T))
    print("    s = {:s}".format(s))
    print("    lam = {:s}".format(lam))
    print("----- point -------------------------------\n")
    print("----- residual ----------------------------")
    print("    obj = {:s}".format(F_1(x, lam).T))
    print("    slack = {:s}".format(F_2(s, lam, mu)))
    print("    constr = {:s}".format(F_3(x, s)))
    print("----- residual ----------------------------")

def visualize(xs, ss, lams, ms, xs_cp, mu):
    ## initialize contours
    delta = 0.025
    x1 = np.arange(-2.0, 2.0, delta)
    x2 = np.arange(-5.0, 10.0, delta)
    levels = range(4,11)
    levels += [15, 20, 25, 30]
    X1, X2 = np.meshgrid(x1,x2)
    Y = fp(X1, X2)

    fig = plt.figure()

    ## initialize path subplot
    plt.subplot(211)

    ## contours
    CS = plt.contour(X1, X2, Y, levels=levels, linewidths=0.5)
    plt.clabel(CS, inline=1, fontsize=10)

    ## path & constraint & full step
    plt.plot([xx[0] for xx in xs], [xx[1] for xx in xs], marker=".", label="cp_ntn", linewidth=0.5, markersize=3)
    plt.plot([xx[0] for xx in xs_cp], [xx[1] for xx in xs_cp], marker=".", label="cp_num", linewidth=0.5, markersize=3)
    plt.plot(x1, (x1 + 0.25)**2/0.75, label="const = 0", linewidth=0.5, color="red")
    # for i in range(len(xs)):
    #     plt.annotate("(s={:s}, lam={:s})".format(ss[i], lams[i]), xy=(xs[i][0], xs[i][1]), textcoords='data')
    plt.legend(fontsize=5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("path")

    ## initialize path subplot
    ax = plt.subplot(212)
    ax.semilogy(ss, label="slacks", linewidth=0.5)
    ax.semilogy(ms, label="merits", linewidth=0.5)
    plt.legend(loc=2, fontsize=5)
    plt.xlabel("iteration")
    plt.ylabel("s or merit value")
    axx = ax.twinx()
    axx.plot(lams, label="duals", color='red', linewidth=0.5)
    plt.legend(loc=1, fontsize=5)
    plt.ylabel("dual")
    plt.title("s, lambda, and merit")

    plt.tight_layout()
    plt.show()
    fig.savefig("output/cp_m{:0.1f}_x1={:0.1f}_x2={:0.1f}.pdf".format(mu, xs[0][0][0], xs[0][1][0]))

def visualize_sims(xss, sss, lamss, mss, xss_cp, mus):
    ## initialize colors
    c = ['orange', 'green', 'blue', 'purple']

    ## initialize contours
    delta = 0.025
    x1 = np.arange(-2.0, 2.0, delta)
    x2 = np.arange(-5.0, 10.0, delta)
    levels = range(4,11)
    levels += [15, 20, 25, 30]
    X1, X2 = np.meshgrid(x1,x2)
    Y = fp(X1, X2)

    fig = plt.figure()

    ## initialize path subplot
    plt.subplot(211)

    ## contours
    CS = plt.contour(X1, X2, Y, levels=levels, linewidths=0.5)
    plt.clabel(CS, inline=1, fontsize=10)

    ## path & constraint & full step
    for i in range(len(xss)):
        xs = xss[i]
        xs_cp = xss_cp[i]
        mu = mus[i]
        plt.plot([xx[0] for xx in xs], [xx[1] for xx in xs], marker=".", linewidth=0.5, markersize=3, color=c[i], alpha=1)
        plt.plot([xx[0] for xx in xs_cp], [xx[1] for xx in xs_cp], marker=".", linewidth=0.5, markersize=3, color=c[i], alpha=0.5)
        plt.plot([], [], label="mu0={:0.1f}".format(mu), color=c[i])
    plt.plot([], [], label="ntn".format(mu), color="black", alpha=1)
    plt.plot([], [], label="tru".format(mu), color="black", alpha=0.5)
    plt.plot(x1, (x1 + 0.25)**2/0.75, label="const = 0", linewidth=0.5, color="red")
    plt.legend(fontsize=5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("path")

    ## initialize path subplot
    ax = plt.subplot(212)
    for i in range(len(xss)):
        ss = sss[i]
        ms = mss[i]
        ax.semilogy(ss, linewidth=0.5, color=c[i])
        ax.semilogy(ms, linewidth=0.5, color=c[i], alpha=0.6)
        ax.plot([], [], label="mu0={:0.1f}".format(mu), color=c[i])
    plt.legend(loc=2, fontsize=5)
    plt.xlabel("iteration")
    plt.ylabel("s or merit value")
    axx = ax.twinx()
    for i in range(len(xss)):
        lams = lamss[i]
        axx.plot(lams, linewidth=0.5, color=c[i], alpha=0.3)
    plt.plot([], [], label="slack".format(mu), color="black", alpha=1)
    plt.plot([], [], label="merit".format(mu), color="black", alpha=0.6)
    plt.plot([], [], label="dual".format(mu), color="black", alpha=0.3)
    plt.legend(loc=1, fontsize=5)
    plt.ylabel("dual")
    plt.title("s, lambda, and merit")

    plt.tight_layout()
    plt.show()
    fig.savefig("output/cp_sim.pdf")
## ===== other =================================================================

## ===== rootfinding ===========================================================
def F_1(x, lam):
    """ obj component in gradient of Lagrangian """
    ## note: not general for m > 1
    return grad_f(x) - lam*grad_g(x)

def F_2(s, lam, mu):
    """ slack component in gradient of Lagrangian """
    e = np.ones([len(s), 1])
    return dmat(s).dot(dmat(lam)).dot(e) - mu*e

def F_3(x, s):
    """ constr component in gradient of Lagrangian """
    return -(g(x) - s)

def F(x, s, lam, mu):
    """
    gradient of Lagrangian (full function for rootfinding)
    """
    ## initialize
    assert(len(s) == len(lam))

    ## compute
    F1 = F_1(x, lam)
    F2 = F_2(s, lam, mu)
    F3 = F_3(x, s)
    out = np.vstack([F1, F2, F3])
    return out

def J_F(x, s, lam, mu):
    """
    Jacobian of Lagrangian (full function for rootfinding)
    """
    ## initialize
    assert(len(s) == len(lam))
    n, m = x.shape[0], s.shape[0]

    ## compute components
    JF11 = hess_f(x) - lam*hess_g(x)
    JF12 = np.zeros([n, m])
    JF13 = -grad_g(x)

    JF21 = np.zeros([m, n])
    JF22 = dmat(lam)
    JF23 = dmat(s)

    JF31 = -grad_g(x).T
    JF32 = np.eye(m)
    JF33 = np.zeros([m, m])

    ## combine components
    out = np.bmat([[JF11, JF12, JF13],
                   [JF21, JF22, JF23],
                   [JF31, JF32, JF33]])
    return out
## ===== rootfinding ===========================================================

## ===== merit / step ==========================================================
def update(x, s, lam, p):
    p = p.reshape(len(p),)
    px = p[0:2]
    px = px.reshape(2,1)
    ps = p[2:3]
    ps = ps.reshape(1,)
    plam = p[3:]
    plam = plam.reshape(1,)
    x += px
    s += ps
    lam += plam
    return x, s, lam

def step(x, s, lam, p):
    # a = la.norm(F_1(x, lam), 2)
    # b = la.norm(F_2(s, lam, mu), 2)
    # c = la.norm(F_3(x, s), 2)
    # d = -s
    # e = -lam
    alpha = 1
    i = 0
    xx, ss, lamlam = update(np.copy(x), np.copy(s), np.copy(lam), np.copy(p))
    while (ss < 0  or lamlam < 0) and i < 1000:
        alpha *= 0.9
        xx, ss, lamlam = update(x, s, lam, alpha*p)
        i += 1
    if i > 1000:
        print("warning")
    return xx, ss, lamlam
## ===== merit / step ==========================================================

## ===== scipy optim ===========================================================
def constraint(x):
    x1, x2 = x[0], x[1]
    return -(x1 + 0.25)**2 + 0.75*x2

def objective(x, mu):
    x1, x2 = x[0], x[1]
    return x1**4 - 2.*(x2 * x1**2) + x2**2 + x1**2 - 2.*x1 + 5. - mu*np.log(constraint(x))

def get_cp(x, mu):
    n = 2
    b = (-10.0, 10.0)
    bnds = (b, b)
    con = {'type': 'ineq', 'fun': constraint}
    cons = ([con])
    solution = minimize(objective, x, args=mu, method='SLSQP',\
                        bounds=bnds, constraints=cons)
    x = solution.x

    # show final objective
    print('Final SSE Objective: ' + str(objective(x, mu)))
    # print solution
    print('Solution')
    print('x1 = ' + str(x[0]))
    print('x2 = ' + str(x[1]))

    return x

## ===== scipy optim ===========================================================
