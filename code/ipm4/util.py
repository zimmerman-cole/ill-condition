import numpy as np
import numpy.linalg as la
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

def visualize(xs, ss, lams):
    ## initialize contours
    delta = 0.025
    x1 = np.arange(-3.0, 3.0, delta)
    x2 = np.arange(-5.0, 10.0, delta)
    levels = range(4,11)
    levels += [15, 20, 25, 30]
    X1, X2 = np.meshgrid(x1,x2)
    Y = fp(X1, X2)

    plt.figure()

    ## initialize path subplot
    plt.subplot(211)

    ## contours
    CS = plt.contour(X1, X2, Y, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)

    ## path
    plt.plot([xx[0] for xx in xs], [xx[1] for xx in xs], marker="X", label="central path")
    # for i in range(len(xs)):
    #     plt.annotate("(s={:s}, lam={:s})".format(ss[i], lams[i]), xy=(xs[i][0], xs[i][1]), textcoords='data')
    plt.legend()
    plt.title("path")

    ## initialize path subplot
    plt.subplot(212)
    plt.plot(ss, label="s")
    plt.plot(lams, label="lam")
    plt.legend()
    plt.title("s and lambda")
    plt.show()
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
## ===== rootfinding ===========================================================
