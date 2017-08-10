import numpy as np
import numpy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from pprint import pprint
from scipy.stats import norm
import matplotlib.pyplot as plt
import util, optimize
from tomo1D import blur as blur
import time


def direct_rxn(X=None, lam=None, B=None, sparse=True):
    n = X.shape[1]
    if B is None:
        B = np.diag(np.ones(n))
        A = X.T.dot(X) + lam*B.T.dot(B)
    A = X.T.dot(X) + lam*B.T.dot(B)
    if sparse:
        R = spsla.spsolve(A, X.T, use_umfpack=True)
    else:
        R = la.solve(A, X.T)
    return R

def direct_solve(Kb=None, R=None, M=None, lam=None, B=None, sb=None, sparse=True):
    MR = M.dot(R)
    print(MR.shape, "MR")
    print(Kb.shape, "Kb")
    Lx = MR.dot(Kb)
    Kx = Lx.dot(MR.T)
    sx = MR.dot(sb)
    if sparse:
        w = spsla.spsolve(Kx,sx)
    else:
        w = la.solve(Kx,sx)
    return w, Kx, sx

def gen_ESI_system(X=None, Kb=None, B=None, M=None, lam=None, sb=None):
    """
    Generates "Equivalent Symmetric Indefinite" LHS and RHS based on III
    """
    m, n = X.shape[0], X.shape[1]
    if B is None: B = sps.eye(n)

    ## intermediate calc
    Z = (X.T.dot(X) + lam*B.T.dot(B))

    ## block LHS
    A11 = X.T.dot(Kb).dot(X)
    A12 = Z.dot(sps.eye(n) - M.T.dot(M))
    A21 = A12.T
    A22 = np.zeros([n,n])
    A = sps.bmat([[A11,A12], [A21,None]])

    ## block RHS
    b1 = X.T.dot(sb)
    b = np.concatenate([b1.reshape(n,), np.zeros(n).reshape(n,)])

    return A, b
