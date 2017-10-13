import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()

def ipm(x0=None, s0=None, y0=None, z0=None, \
        mu0=None, sigma=None, tau=None, \
        niter=10):

    ## set initial parameters
    if x0 is None: x0 = np.array([-1.71, 1.59, 1.82, -0.763, -0.763])
    if s0 is None: s0 = np.array([1., 1.])
    if y0 is None: y0 = np.array([1., 1., 1.])
    if z0 is None: z0 = np.array([1., 1.])
    if mu0 is None: mu0 = 2.
    if sigma is None: sigma = 0.5     # python only
    if tau is None: tau = 0.995       # python only
    err = 1e6                         # python only

    ## prepare for iterations
    x = x0
    s = s0
    y = y0
    z = z0
    mu = mu0                           # python only

    ## solver interior Newton systems
    for k in range(niter):
        print("==== iteration: {:d} ================================".format(k))

        ## prepare current iterate to MATLAB
        x = matlab.double([q for q in x])
        s = matlab.double([q for q in s])
        y = matlab.double([q for q in y])
        z = matlab.double([q for q in z])
        inner = 0

        ## determine step
        while err > mu:
            ## setup and solve Newton System
            out = eng.ntsys(x, s, y, z, mu, nargout=4)
            [J, h, p, err] = [np.array(o) for o in out]
            ## reformatting
            p_x = p[0:5]
            p_s = p[5:7]
            p_y = p[7:10]
            p_z = p[10:12]
            x = np.array(x).reshape(5,1)
            s = np.array(s).reshape(2,1)
            y = np.array(y).reshape(3,1)
            z = np.array(z).reshape(2,1)
            ## step length
            a = np.linspace(0.,1., 100)
            a_s = a[max(np.where(s[0] + a*p_s >= (1.-tau)*s[0])[0])]
            a_z = a[max(np.where(z[0] + a*p_z[0][0] >= (1.-tau)*z)[0])]
            ## updates
            x += a_s*p_x
            s += a_s*p_s[0]
            y += a_z*p_y
            z += a_z*p_z[0]
            ## prepare current information for MATLAB
            x = matlab.double([float(q) for q in x])
            s = matlab.double([float(q) for q in s])
            y = matlab.double([float(q) for q in y])
            z = matlab.double([float(q) for q in z])
            ## compute error
            out = eng.ntsys(x, s, y, z, mu, nargout=4)
            [_, _, _, err] = [np.array(o) for o in out]
            ## tracking
            inner += 1
            print(x)
            print(e)
            print(inner)

        ## adjust mu
        mu *= sigma

    return x

if __name__ == "__main__":
    x = ipm()
    print(x)
