import numpy as np
import scipy.sparse as sps
from pprint import pprint
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

def gen_f(n):
    f_impulse = x_true = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    name = "f_impulse"+"_"+str(n)
    np.save(name,f_impulse)
    plt.plot(range(n),f_impulse)
    plt.show()

# filename = "f_impulse_100.npy"
# f = np.load(filename)
# plt.plot(f)
# plt.show()

def template_1d(sigma=None, t=0, debug=False):
    """
    Computes discretized template for 1D Gaussian Blur
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
    x = norm(0,sigma)

    ## construct sliding pixel template
    template_inds = range(-t,t+1)
    template = [(t-0.5, t+0.5) for t in template_inds]

    ## integrate over midpoints to get weights
    template = [x.cdf(t[1]) - x.cdf(t[0]) for t in template]

    ## normalize
    sumt = sum(template)
    template = [t/sumt for t in template]

    ## return
    return template, template_inds

def row_k(k=0, template=None, template_inds=None, n=None, sparse=True, debug=False):
    """
    Returns a row of 1D Gaussian Blur
    Args
        - k             :  central pixel
        - template      :  window and Gaussian approximation
        - template_inds :  pixel indices
        - sparse        :  store rows as sparse csr_matrix
    Return
        - r_k           :  row filter to apply to f for k-th pixel
    """
    if k is None:
        print("specify `k` in row_k")
        sys.exit(0)

    if n is None:
        print("specify `n` in row_k")
        sys.exit(0)

    ## compute window size from template
    t = int(np.floor(len(template)/2))

    ## initialize row with all zeros
    r_k = np.zeros(n)

    ## fill in template
    inds = [(t+k)%n for t in template_inds]
    r_k[inds] = template

    ## sparse representation
    if sparse:
        r_k = sps.csr_matrix(r_k)

    ## debug
    if debug:
        rr_k = r_k.toarray()
        # rr_k = [round(r,3) for r in rr_k]
        rr_k = np.round_(rr_k,3)
        print(rr_k,)
        print("type r_k",type(rr_k))
        print("")

    ## return
    return r_k

def fwdblur_oeprator_1d(n=None, sigma=3, t=3, sparse=True, plot=False, debug=False):
    """
    Returns an n x n np.array
    Args
        - n      :  pixels of original image
        - sparse :  creates X as sparse csr_matrix
    Returns
        - X      :  n x n (Gaussian) blur operator
    """
    if n is None:
        print("specify `n`")
        sys.exit(0)

    if sigma is None:
        print("specify `sigma`")
        sys.exit(0)

    if t is None:
        print("specify `t`")
        sys.exit(0)


    ## compute template
    template, template_inds = template_1d(sigma=sigma, t=t, debug=debug)

    ## plot template
    if plot:
        fig, ax = plt.subplots()
        ax.plot(template_inds, template, color='r')
        plt.xlabel("pixel")
        plt.ylabel("approx weight")
        ax.set_xticks(template_inds)
        plt.show()

    ## construct sparse
    if sparse:
        X = row_k(k=0, template=template, template_inds=template_inds, n=n, sparse=True, debug=debug)
        for k in range(1, n):
            r_k = row_k(k=k, template=template, template_inds=template_inds, n=n, sparse=True, debug=debug)
            X = sps.vstack([X, r_k])
    else:
        X = row_k(k=0, template=template, template_inds=template_inds, n=n, sparse=False, debug=debug)
        for k in range(1, n):
            r_k = row_k(k=k, template=template, template_inds=template_inds, n=n, sparse=False, debug=debug)
            X = np.vstack([X, r_k])

    if debug:
        XX = X.toarray()
        XX = np.round_(XX,3)
        print(XX.shape)
        pprint(XX)
    ## return
    return X

def test_symm(X, d=8):
    """
    Test symmetry to `d` digits of precision
    """
    X = X.toarray()
    T = X.T - X
    T = np.round_(T,d)
    if abs(sum(sum(T))) < 10**-d:
        return True
    else:
        return False

X = fwdblur_oeprator_1d(n=11, plot=False, debug=True)
print('===========')
print(test_symm(X))
