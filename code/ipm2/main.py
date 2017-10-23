import numpy as np
import numpy.linalg as la
import util

def ipm(x0=None, s0=None, z0=None, y0=None, mu=None, tau=None, K=2, debug=True):
    ## initialize
    if x0 is None: x0 = np.array([2., 3.])
    if s0 is None: s0 = np.array([6.,13.])
    if z0 is None: z0 = np.array([-0.4053, -0.3052])
    if mu is None: mu = 10.
    if tau is None: tau = 0.99
    x, s, z, y= x0, s0, z0, y0
    x = x.reshape(2,1)
    s = s.reshape(2,1)
    z = z.reshape(2,1)
    err = 1e6

    for k in range(K):
        print("=== iter {:d} ==================================================================".format(k))
        if debug:
            print("f(x): {:s}\n".format(util.f0(x)))
            print("x: {:s}\n".format(x.T))
            print("s: {:s}\n".format(s.T))
            print("z: {:s}\n".format(z.T))

        ## setup system
        print("    --- system ---------------------------------------------------------------\n")
        JF = util.JF(x, s, z, mu, debug=debug)
        F = util.F(x, s, z, mu, debug=debug)

        ## solve system
        p = la.solve(JF, F)
        p_x, p_s, p_z = np.split(p, 3)
        print("    --- system ---------------------------------------------------------------\n")

        ## update
        x += p_x
        s += p_s
        z += p_z

        if debug:
            print("~~~ computed direction! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("p_x: {:s}\n".format(p_x.T))
            print("p_s: {:s}\n".format(p_s.T))
            print("p_z: {:s}\n".format(p_z.T))
            print("~~~ updated variables! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("x: {:s}\n".format(x.T))
            print("s: {:s}\n".format(s.T))
            print("z: {:s}\n".format(z.T))
            print("f(x): {:s}".format(util.f0(x)))
            print("~~~ compare to optimal! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("x*:   {:s}".format([4.3198, 1.1575]))
            print("s*:   {:s}".format([5., 20.]))
            print("z*:   {:s}".format([2.2470, 0.1671]))
        print("=== iter {:d} ==================================================================\n".format(k))
    return x

if __name__ == '__main__':
    ipm()
