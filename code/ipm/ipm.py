import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import util

def ipm(x0=None, s0=None, y0=None, z0=None, \
        mu0=None, sigma=None, tau=None, \
        K=10):

    ## set initial parameters
    if x0 is None: x0 = [-1.71, 1.59, 1.82, -0.763, -0.763]
    if s0 is None: s0 = [1., 1.]
    if y0 is None: y0 = [1., 1., 1.]
    if z0 is None: z0 = [1., 1.]
    if mu0 is None: mu0 = 2.
    if sigma is None: sigma = 0.5
    if tau is None: tau = 0.995
    err = 1e6

    ## prepare for iterations
    x = x0
    s = s0
    y = y0
    z = z0
    mu = mu0

    ## solve interior Newton systems
    for k in range(K):
        print("==== iteration: {:d} ================================".format(k))
        inner = 0

        ## determine step
        while err > mu:
            ## tracking
            print('x = %s' % (x))
            print('s = %s' % (s))
            print('y = %s' % (y))
            print('z = %s' % (z))

            ## setup and solve Newton System
            h, J = util.nt_sys(x, s, y, z, mu)
            p = la.solve(J,-h)
            p_x = p[0:5]
            p_s = p[5:7]
            p_y = p[7:10]
            p_z = p[10:12]

            ## step length
            a = np.linspace(0.,1., 100)
            a_s = opt.minimize(util.find_step_len, x0=1., args=(p_s, s, tau), method="L-BFGS-B")
            a_z = opt.minimize(util.find_step_len, x0=1., args=(p_z, s, tau), method="L-BFGS-B")
            a_s = a_s.x
            a_z = a_z.x

            ## updates
            x += a_s*p_x
            s += a_s*p_s[0]
            y += a_z*p_y
            z += a_z*p_z[0]

            ## compute error
            err = util.calc_err(h)

            ## tracking
            inner += 1
            print(x)
            print(inner)

        ## adjust mu
        mu *= sigma

    return x

if __name__ == "__main__":
    x = ipm()
    print(x)
