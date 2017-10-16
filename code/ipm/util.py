import numpy as np
import numpy.linalg as la

def ml_adj(x):
    # print(x, "x orig: ", type(x))
    if type(x) == np.ndarray:
        x = np.concatenate((np.array([np.nan]),x),axis=0)
    else:
        x = np.array([np.nan] + x)  # adjust numbering for MATLAB notation
    # print(x, "x adj: ", type(x))
    return x

def f(x):
    ## MATLAB adjust
    x = ml_adj(x)

    ep = np.exp(np.prod(x[1:]))
    f = ep - 0.5*(x[1]**3 + x[2]**3 + 1)**2
    return f

def g(x):
    ## MATLAB adjust
    x = ml_adj(x)

    ep = np.exp(np.prod(x[1:]))
    g = [ \
        x[2]*x[3]*x[4]*x[5]*ep - 3*x[1]**2 * (x[1]**3 + x[2]**3 + 1),
        x[3]*x[4]*x[5]*x[1]*ep - 3*x[2]**2 * (x[1]**3 + x[2]**3 + 1),
        x[4]*x[5]*x[1]*x[2]*ep,
        x[5]*x[1]*x[2]*x[3]*ep,
        x[1]*x[2]*x[3]*x[4]*ep
        ]
    g = np.array(g)
    g = g.reshape(5,1)
    return g

def H(x):
    ## MATLAB adjust
    x = ml_adj(x)

    ep = np.exp(np.prod(x[1:]))
    H = \
    [
        [x[2]**2*x[3]**2*x[4]**2*x[5]**2*ep-9*x[1]**4-6*(x[1]**3+x[2]**3+1)*x[1],
            x[3]*x[4]*x[5]*ep+x[2]*x[3]**2*x[4]**2*x[5]**2*x[1]*ep-9*x[2]**2*x[1]**2,
            x[2]*x[4]*x[5]*ep+x[2]**2*x[3]*x[4]**2*x[5]**2*x[1]*ep,
            x[2]*x[3]*x[5]*ep+x[2]**2*x[3]**2*x[4]*x[5]**2*x[1]*ep,
            x[2]*x[3]*x[4]*ep+x[2]**2*x[3]**2*x[4]**2*x[5]*x[1]*ep
        ],
        [x[3]*x[4]*x[5]*ep+x[2]*x[3]**2*x[4]**2*x[5]**2*x[1]*ep-9*x[2]**2*x[1]**2,
            x[1]**2*x[3]**2*x[4]**2*x[5]**2*ep-9*x[2]**4-6*(x[1]**3+x[2]**3+1)*x[2],
            x[1]*x[4]*x[5]*ep+x[1]**2*x[3]*x[4]**2*x[5]**2*x[2]*ep,
            x[1]*x[3]*x[5]*ep+x[1]**2*x[3]**2*x[4]*x[5]**2*x[2]*ep,
            x[1]*x[3]*x[4]*ep+x[1]**2*x[3]**2*x[4]**2*x[5]*x[2]*ep
        ],
        [x[2]*x[4]*x[5]*ep+x[2]**2*x[3]*x[4]**2*x[5]**2*x[1]*ep,
            x[1]*x[4]*x[5]*ep+x[1]**2*x[3]*x[4]**2*x[5]**2*x[2]*ep,
            x[1]**2*x[2]**2*x[4]**2*x[5]**2*ep,
            x[1]*x[2]*x[5]*ep+x[1]**2*x[2]**2*x[4]*x[5]**2*x[3]*ep,
            x[1]*x[2]*x[4]*ep+x[1]**2*x[2]**2*x[4]**2*x[5]*x[3]*ep
        ],
        [x[2]*x[3]*x[5]*ep+x[2]**2*x[3]**2*x[4]*x[5]**2*x[1]*ep,
            x[1]*x[3]*x[5]*ep+x[1]**2*x[3]**2*x[4]*x[5]**2*x[2]*ep,
            x[1]*x[2]*x[5]*ep+x[1]**2*x[2]**2*x[4]*x[5]**2*x[3]*ep,
            x[1]**2*x[2]**2*x[3]**2*x[5]**2*ep,
            x[1]*x[2]*x[3]*ep+x[1]**2*x[2]**2*x[3]**2*x[5]*x[4]*ep
        ],
        [x[2]*x[3]*x[4]*ep+x[2]**2*x[3]**2*x[4]**2*x[5]*x[1]*ep,
            x[1]*x[3]*x[4]*ep+x[1]**2*x[3]**2*x[4]**2*x[5]*x[2]*ep,
            x[1]*x[2]*x[4]*ep+x[1]**2*x[2]**2*x[4]**2*x[5]*x[3]*ep,
            x[1]*x[2]*x[3]*ep+x[1]**2*x[2]**2*x[3]**2*x[5]*x[4]*ep,
            x[1]**2*x[2]**2*x[3]**2*x[4]**2*ep
        ]
    ]
    H = np.array(H)
    return H

def c_E(x):
    ## MATLAB adjust
    x = ml_adj(x)

    c_E = \
    [
       x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2,
       x[2]*x[3] - 5*x[4]*x[5],
       x[1]**3 + x[2]**3 + 1
    ]
    c_E = np.array(c_E)
    c_E = c_E.reshape(3,1)
    return c_E

def c_I(x, s):
    ## MATLAB adjust
    x = ml_adj(x)
    s = ml_adj(s)

    c_I = \
    [
       x[2] - s[1],
       x[3] - s[2]
    ]
    c_I = np.array(c_I)
    c_I = c_I.reshape(2,1)
    return c_I

def A_E(x):
    ## MATLAB adjust
    x = ml_adj(x)

    A_E = \
    [
       [2.*x[1], 2.*x[2], 2.*x[3], 2.*x[4], 2.*x[5]],
       [0, x[3], x[2], -5.*x[5], -5.*x[4]],
       [3.*x[1]**2, 3.*x[2]**2, 0, 0, 0]
    ]
    A_E = np.array(A_E)
    A_E = A_E.reshape(3,5)
    return A_E

def A_I():
    A_I = \
    [
       [0, 1., 0, 0, 0],
       [0, 0, 1., 0, 0]
    ];
    A_I = np.array(A_I)
    A_I = A_I.reshape(2,5)
    return A_I


def nt_sys(x, s, y, z, mu):
    """
    generates Newton system
    input:
    - x  := 5x1 vector
    - s  := 2x1 vector
    - y  := 3x1 vector
    - mu := 1x1 vector
    """
    ## setup
    e = np.array([1.,1.]).reshape(2,1)
    S = [[s[0], 0],
         [0, s[1]]]
    S = np.array(S)
    Z = [[z[0], 0],
         [0, z[1]]]
    Z = np.array(Z)
    y = np.array(y).reshape(3,1)
    z = np.array(z).reshape(2,1)

    ## compute grad, hess, constrs
    ff, gg, HH = f(x), g(x), H(x)
    cc_E, cc_I, AA_E, AA_I = c_E(x), c_I(x, s), A_E(x), A_I()

    ## setup system
    h = np.bmat([
       [gg - AA_E.T.dot(y) - AA_I.T.dot(z)],   # 5x1
       [S.dot(z) - mu*e],                      # 2x1
       [cc_E],                                 # 3x1
       [cc_I]                                  # 2x1
    ])
    h = np.array(h)
    h = h.reshape(12,1)

    J = np.bmat([
       [HH, np.zeros([5,2]), -AA_E.T, -AA_I.T],
       [np.zeros([2,5]), Z, np.zeros([2,3]), S],
       [AA_E, np.zeros([3,2]), np.zeros([3,3]), np.zeros([3,2])],
       [AA_I, -np.eye(2), np.zeros([2,3]), np.zeros([2,2])]
    ])

    return h, J

def calc_err(h):
    err = max([la.norm(h[0:5],2), la.norm(h[5:7],2), la.norm(h[7:10],2), la.norm(h[10:12],2)])
    return err

def find_step_len(alpha, p, s, tau):
    p = np.array(p)
    s = np.array(s)
    print("alpha = ", alpha, type(alpha))
    # print("p = ", p, type(p))
    # print("s = ", s, type(s))
    # print("tau = ", tau, type(tau))
    a = np.prod(tau*s + alpha*p >= 0)
    b1 = (alpha>=0.)
    b2 = (alpha<=1.)
    constr = a*b1*b2
    f = -1.*(alpha + constr)
    print(f)
    return f
