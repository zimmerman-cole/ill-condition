import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def gen_f(n):
    f_impulse = x_true = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    name = "f_impulse"+"_"+str(n)
    np.save(name,f_impulse)
    plt.plot(range(n),f_impulse)
    plt.show()

filename = "f_impulse_100.npy"
f = np.load(filename)
plt.plot(f)
plt.show()



def template_1d(sigma=None, t=0):
    """
    Template for Gaussian Blur of 1D image
    Args
        - f     :  image
        - sigma :  standard deviation
        - t     :  one-sided pixel window (scale-space)
                   t = 0       ==> original image (impulse response)
                   t = sigma^2 ==>
    Return
        - A     :  forward blur matrix
    """

    ## construct RV for integration
    x = norm.(0,sigma)

    ## construct sliding pixel template
    template_inds = arange(-t,t+1)
    template = [(t-0.5, t+0.5) for t in template]

    ## integrate over midpoints to get weights
    template = [x.cdf(t[1]) - x.cdf(t[0]) for t in template]

    ## normalize
    sumt = sum(template)
    template = [t/sumt for t in template]

    ## return
    return template

def row1(template, n)
