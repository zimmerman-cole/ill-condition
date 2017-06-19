import numpy as np
import numpy.linalg as la
import scipy as sp
import random 

from scipy.optimize import minimize

## generating ill-conditioned matrix A
lo = -1000  # low range of random integers
hi = 1000   # high range of random integers
n = 100     # dimension (square) of A
line = [random.randint(lo,hi) for i in range(n)]
A = np.vander(line)

## generate true solution
x_true = [random.randint(lo,hi) for i in range(n)]

## generate data
b = np.dot(A,x_true)

## initial (bad, random) guess
x_init = [np.random.normal() for i in range(n)]

## ||Ax - b||
def linear_diff(x,A,b):
    return la.norm(np.dot(A,x)-b)

## various methods for minimizing ||Ax - b|| based on x_init
opts = ['Nelder-Mead','CG','BFGS','Newton-CG','L-BFGS-B']

for opt in opts:
    res = sp.optimize.minimize(linear_diff,x_init,args=(A,b),method=opt)
    print(res.success)
    print(res.message)
