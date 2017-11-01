import numpy as np
import numpy.linalg as la
import util

def ipm(x=None, s=None, lam=None, mu=None, K=10, debug=True):
    x, s, lam, mu = util.parse_input(x, s, lam, mu)
    for k in range(K):
        print("===== start iter {:d} =============================".format(k))
        util.merit_status(x, s, lam, mu)
        ## solve
        F = util.F(x, s, lam, mu)
        J_F = util.J_F(x, s, lam, mu)
        p = la.solve(J_F, -F)
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
        print("===== end iter {:d} ===============================\n".format(k))

if __name__ == '__main__':
    x = [-1., 4.]
    s = 2.4375
    lam = 2
    mu = 5
    ipm(x=x, s=s, lam=lam, mu=mu)
