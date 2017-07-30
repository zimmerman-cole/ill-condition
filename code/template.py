import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from pprint import pprint
from scipy.stats import norm
import matplotlib.pyplot as plt
import util, optimize
from tomo1D import blur as blur

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
