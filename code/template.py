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
    if B is None:
        B = np.diag(np.ones())
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

def gen_ESI_system(m=None, n=None, k=None, X=None, B=None, M=None, lam=none):
    """
    Generates "Equivalent Symmetric Indefinite" LHS and RHS based on III
    """
    ## intermediate calc
    Z = (X.T.dot(X) + lam*B.T.dot(B))

    ## block LHS
    A11 = X.T.dot(Kb).dot(X)
    A12 = Z.dot(iden(n) - M.T.dot(M))
    A21 = A12.T
    A22 = np.zeros([n,n])
    A = sps.bmat([[A11,A12], [A21,None]])

    ## block RHS
    b1 = X.T.dot(sb)
    b = np.concatenate([b1, np.zeros(n)])

    return A, b
