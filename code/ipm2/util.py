import numpy as np
import numpy.linalg as la

def f0(x):
    return x[1] * (5.+x[0])

def grad_f0(x):
    g = np.zeros([2,1])
    g[0] = x[1]
    g[1] = 5. + x[0]
    return g

def grad2_f0(x):
    H = np.zeros([2,2])
    H[0,0] = 0
    H[0,1] = 1.
    H[1,0] = 1.
    H[1,1] = 0
    return H

def grad_f1(x):
    g = np.zeros([2,1])
    g[0] = x[1]
    g[1] = x[0]
    return g

def grad2_f1(x):
    H = np.zeros([2,2])
    H[0,0] = 0
    H[0,1] = -1.
    H[1,0] = -1.
    H[1,1] = 0
    return H

def grad_f2(x):
    g = np.zeros([2,1])
    g[0] = 2.*x[0]
    g[1] = 2.*x[1]
    return g

def grad2_f2(x):
    H = np.zeros([2,2])
    H[0,0] = -2.
    H[0,1] = 0
    H[1,0] = 0
    H[1,1] = -2.
    return H

def grad_lag(x, z):
    return grad_f0(x) - z[0]*grad_f1(x) - z[1]*grad_f2(x)

def SS(s):
    S = np.zeros([2,2])
    S[0,0] = s[0]
    S[1,1] = s[1]
    return S

def ZZ(z):
    Z = np.zeros([2,2])
    Z[0,0] = z[0]
    Z[1,1] = z[1]
    return Z

def slack(s, z, mu):
    ## setup
    S = SS(s)
    Z = ZZ(z)
    e = np.ones([2,1])
    return S.dot(z).reshape(2,1) - mu*e

def cons(x, s):
    con1 = x[0]*x[1] - 5. - s[0]
    con2 = 20. - x[0]**2 - x[1]**2 - s[1]
    return np.array([con1, con2]).reshape(2,1)

def F(x, s, z, mu, componentwise=False, debug=False):
    ## compute components
    lag = grad_lag(x, z)
    slk = slack(s, z, mu)
    con = cons(x, s)

    if debug > 1:
        u = [lag, slk, con]
        print([v.shape for v in u])

    out = np.vstack([lag,
                    slk,
                    con])

    if debug:
        print("F:")
        print(out)

    if componentwise:
        return lag, slk, con, out
    else:
        return out

def JF(x, s, z, mu, debug=False):
    ## setup
    S = np.zeros([2,2])
    S[0,0] = s[0]
    S[1,1] = s[1]
    Z = np.zeros([2,2])
    Z[0,0] = z[0]
    Z[1,1] = z[1]
    e = np.ones([2,1])

    ## compute components
    j11 = grad2_f0(x) - z[0]*grad2_f1(x) - z[1]*grad2_f2(x)  # wrt x1, x2
    j12 = np.zeros([2,2])                                    # wrt s1, s2
    j13 = np.hstack([grad_f1(x), grad_f2(x)])                # wrt z1, z2 (not sure if .T is right)

    j21 = np.zeros([2,2])                                    # wrt x1, x2
    j22 = Z                                                  # wrt s1, s2
    j23 = S                                                  # wrt z1, z2

    j31 = np.vstack([grad_f1(x).T, grad_f2(x).T])            # wrt x1, x2
    j32 = -np.eye(2)                                         # wrt s1, s2
    j33 = np.zeros([2,2])                                    # wrt z1, z2

    if debug > 1:
        jj = [[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]]
        print("shapes of components of JF")
        print([[v.shape for v in u] for u in jj])
        print()

    ## combine components
    out = np.bmat([[j11, j12, j13],
                   [j21, j22, j23],
                   [j31, j32, j33]])

    if debug:
        print("JF:")
        print(out)

    return out

def calc_err(x, s, z, mu, debug=False):
    lag, slk, con, full_errs = F(x, s, z, mu, componentwise=True)
    a, b, c = la.norm(lag, 2), la.norm(slk, 2), la.norm(con, 2)
    if debug:
        print("lagrangian error: {:f}".format(a))
        print("complementarity error: {:f}".format(b))
        print("constraint error: {:f}".format(c))
    err = max([a, b, c])
    return err

def step(p, s, tau, factor=0.99, debug=False):
    alpha = 1.
    p = np.array(p)
    s = np.array(s)
    i = 0
    while True:
        if debug:
            print("        +++ steplen iteration {:d} ++++++++++++++++++++++++++\n".format(i))
            print("alpha = {:f}".format(alpha))
        d = tau*s + alpha*p < 0
        if d[0] >= 0 and d[1] >= 0:
            return alpha
        else:
            alpha *= factor
            i += 1
