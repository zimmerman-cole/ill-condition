import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
import scipy.sparse as sps
from pprint import pprint
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

def gen_f(n, save=False, plot=True):
    f_impulse = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    if save:
        name = "f_impulse"+"_"+str(n)
        np.save(name,f_impulse)
    if plot:
        plt.plot(range(n),f_impulse)
        plt.title("f_impulse image")
        plt.show()
    return(f_impulse)

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
        if sparse:
            rr_k = r_k.toarray()
        else:
            rr_k = r_k
        rr_k_round = np.round_(rr_k,3)
        print(rr_k,)
        print("type r_k",type(rr_k))
        print("")

    ## return
    return r_k

def fwdblur_operator_1d(n=None, sigma=3, t=10, sparse=True, plot=False, debug=False):
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
        if sparse:
            XX = X.toarray()
        else:
            XX = X
        XX_round = np.round_(XX,3)
        print(XX.shape)
        pprint(XX_round)

        r0 = row_k(k=0, template=template, template_inds=template_inds, n=n, sparse=False, debug=False)
        XXX = spla.circulant(r0)
        print(np.round_(XXX-XX,3))
        print(np.sum(np.round_(XXX-XX,3)))
    ## return
    return X

def test_symm(X, d=8):
    """
    Test symmetry to `d` digits of precision
    """
    if sps.issparse(X):
        X = X.toarray()
    T = X.T - X
    T = np.round_(T,d)
    if abs(sum(sum(T))) < 10**-d:
        return True
    else:
        return False

def test_dft(X, d=8):
    """
    Tests whether DFT diagonalizes X to `d` digits of precision
    """
    F = np.asmatrix(np.fft.fft(X))
    Finv = la.inv(F)
    T = Finv.dot(X).dot(F)
    T = np.round_(T,d)
    if T.trace() - np.sum(T) < 10**-d:
        return True
    else:
        return False

if __name__ == "__main__":
    X = fwdblur_operator_1d(n=11, sigma=3, t=3, plot=True, debug=True)
    print('===========')
    print(test_symm(X))

    X = fwdblur_operator_1d(n=11, sigma=3, t=10, plot=True, debug=True)
    print('===========')
    print(test_symm(X))

    X = fwdblur_operator_1d(n=100, sigma=3, t=10, plot=True, debug=True)
    print(type(X))
    print('===========')
    print(test_symm(X))
