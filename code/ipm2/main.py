import numpy as np
import numpy.linalg as la
import util

def ipm(x0=None, s0=None, z0=None, y0=None, mu=None, tau=None, K=20, debug=True):
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
        i = 0
        print("=== iter {:d} ===================================================\n".format(k))
        JF = util.JF(x, s, z, mu, debug=debug)
        F = util.F(x, s, z, mu, debug=debug)
        p = la.solve(JF, -F)
        p_x, p_s, p_z = np.split(p, 3)

        ## line search
        alpha = 1.
        new_err = err
        while new_err >= err:
            print("    --- inner iteration: {:d} ---------------------------\n".format(i))
            xx = x + alpha*p_x
            ss = s + alpha*p_s
            zz = z + alpha*p_z
            new_err = util.calc_err(xx, ss, zz, mu, debug=debug)
            i += 1
            alpha *= tau

        if debug:
            print("p:")
            print(p)

        if debug > 1:
            print("shape of JF: ({:d},{:d})".format(JF.shape[0], JF.shape[1]))
            print("shape of F: ({:d},{:d})".format(F.shape[0], F.shape[1]))
            print("shape of p: ({:d},{:d})".format(p.shape[0], p.shape[1]))


        if debug > 1:
            print("shape of p: ({:d},{:d})".format(p.shape[0], p.shape[1]))
            print("shape of p_x: ({:d},{:d})".format(p_x.shape[0], p_x.shape[1]))
            print("shape of p_s: ({:d},{:d})".format(p_s.shape[0], p_s.shape[1]))
            print("shape of p_z: ({:d},{:d})".format(p_z.shape[0], p_z.shape[1]))

        ## updates
        x += alpha*p_x.reshape(2,1)
        s += alpha*p_s.reshape(2,1)
        z += alpha*p_z.reshape(2,1)
        err = util.calc_err(x, s, z, mu, debug=debug)

        if debug:
            print("error: {:f}".format(err))
            print

        if err < mu:
            if debug:
                print("x: {:s}".format(x))
                print("f(x): {:s}".format(util.f0(x)))

            return x
    if debug:
        print("x:    {:s}".format(x))
        print("f(x): {:s}".format(util.f0(x)))
        print("s1:   {:s}".format(s[0]))
        print("s2:   {:s}".format(s[1]))
        print("z1:   {:s}".format(z[0]))
        print("z2:   {:s}\n".format(z[1]))
        print('___________________________')
        print("x*:   {:s}".format([4.3198, 1.1575]))
        print("s*:   {:s}".format([5., 20.]))
        print("z*:   {:s}".format([2.2470, 0.1671]))

    return x

if __name__ == '__main__':
    ipm()
