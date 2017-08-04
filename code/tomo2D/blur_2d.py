import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.linalg as spla
import scipy.sparse as sps
from pprint import pprint
from scipy.stats import norm
import matplotlib.pyplot as plt

import os, sys
sys.path.append('..')
from tomo1D import blur_1d as blur_1d

np.set_printoptions(linewidth=200)

def gen_f_rect(n_1, n_2, levels=3, save=False, plot=True):
    """
    Generates rectangular image f with `levels` intensities
    Args:
        - n_1: n rows
        - n_2: n cols
        - levels: number of levels (< min(n_1/2, n_2/2))
    Returns:
        - f_rect: image with `levels` of intensities
    """
    f_rect = np.zeros([n_1,n_2])
    step_1 = int(np.floor(n_1/2.)/(levels+1))
    step_2 = int(np.floor(n_2/2.)/(levels+1))
    c_1 = int(np.floor(n_1/2.))
    c_2 = int(np.floor(n_2/2.))

    if levels >= min(c_1,c_2):
        print("too many levels for pixel resolution; reduce levels")
        raise

    for l in range(levels+1,1,-1):
        i = c_1 - l*step_1
        j = c_2 - l*step_2
        f_rect[(i+1):-(i+1), (j+1):-(j+1)] += l

    if save:
        name = "f_rect"+"_"+str(n_1)+"_"+str(n_2)+"_"+str(levels)
        np.save(name,f_rect)
    if plot:
        plt.imshow(f_rect)
        plt.title("f_rect image")
        plt.show()
    return(f_rect)

# f = gen_f_rect(100, 200, levels=10, save=True)

## row for X_row operator
def row_k(k=0, n_1=None, n_2=None, template=None, template_inds=None, sparse=True, debug=False):
    """
    Returns a row of 1D Gaussian Blur
    Args
        - k             :  central pixel
        - n_1           :  number of rows
        - n_2           :  number of cols
        - template      :  window and Gaussian approximation
        - template_inds :  pixel indices
        - sparse        :  store rows as sparse csr_matrix
    Return
        - r_k           :  row filter to apply to f for k-th pixel
    """
    if k is None:
        print("specify `k` in row_k")
        sys.exit(0)

    if n_1 is None:
        print("specify `n_1` in row_k")
        sys.exit(0)

    ## compute window size from template
    t = int(np.floor(len(template)/2))

    ## initialize row with all zeros
    r_k = np.zeros(n_1 * n_2)

    ## fill in template
    inds = [(t*n_1+k)%(n_2*n_1) for t in template_inds]
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

def fwdblur_operator_2d(n_1=10, n_2=20, sigma=3, t=10, sparse=True, plot=False, debug=False):

    ## COL BLUR - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## define block
    B_col = blur_1d.fwdblur_operator_1d(n=n_1, sigma=sigma, t=t, sparse=sparse, plot=plot, debug=debug)

    print(type(B_col),"B_col type")

    X_col = B_col
    if sparse:
        for col in range(n_2-1):
            X_col = sps.block_diag((X_col,B_col))
    else:
        for col in range(n_2-1):
            X_col = sla.block_diag((X_col,B_col))

    ## ROW BLUR - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## define block
    template, template_inds = blur_1d.template_1d(sigma=sigma, t=t, debug=debug)
    if sparse:
        X_row = row_k(k=0, n_1=n_1, n_2=n_2, template=template, template_inds=template_inds, sparse=True, debug=debug)
        for k in range(1, n_1*n_2):
            r_k = row_k(k=k, n_1=n_1, n_2=n_2, template=template, template_inds=template_inds, sparse=True, debug=debug)
            X_row = sps.vstack([X_row, r_k])
    else:
        X_row = row_k(k=0, n_1=n_1, n_2=n_2, template=template, template_inds=template_inds, sparse=False, debug=debug)
        for k in range(1, n_1*n_2):
            r_k = row_k(k=k, n_1=n_1, n_2=n_2, template=template, template_inds=template_inds, sparse=False, debug=debug)
            X_row = np.vstack([X_row, r_k])

    return X_col, X_row

def example(n_1=20, n_2=50, sigma=5, t=8):
    f = gen_f_rect(n_1=n_1, n_2=n_2, levels=3, plot=True)

    X_col, X_row = fwdblur_operator_2d(n_1=n_1, n_2=n_2, sigma=sigma, t=t)

    fig,ax = plt.subplots(1,3)

    ## COLUMN
    _f_ = f.flatten("F").reshape(n_1*n_2,1)
    print("dim X_col: ", X_col.toarray().shape)
    print("dim _f_: ", _f_.shape)
    _f_colblur = X_col.dot(_f_)
    f_colblur = _f_colblur.reshape(n_1,n_2,order="F")
    plt.subplot(1,3,1)
    plt.imshow(f_colblur)

    ## ROW
    _f_ = f.flatten("F").reshape(n_1*n_2,1)
    print("dim X_row: ", X_row.toarray().shape)
    print("dim _f_: ", _f_.shape)
    _f_rowblur = X_row.dot(_f_)
    f_rowblur = _f_rowblur.reshape(n_1,n_2,order="F")
    plt.subplot(1,3,2)
    plt.imshow(f_rowblur)

    ## BOTH
    _f_blur = X_row.dot(_f_colblur)
    print("dim X_col: ", X_col.toarray().shape)
    print("dim _f_: ", _f_.shape)
    f_blur = _f_blur.reshape(n_1,n_2,order="F")
    plt.subplot(1,3,3)
    plt.imshow(f_blur)
    plt.show()

if __name__ == "__main__":
    example()
