import numpy as np
import numpy.linalg as la
import scipy as sp
import random 
from util import mat_from_cond
from scipy.optimize import minimize

### formulate problem ###
# A \in R^[m x n]
# x \in R^n
# b \in R^m

## generating ill-conditioned matrix A
m = 50
n = 50
A = mat_from_cond(1,m,n)  # args = cond_num, m=50, n=50, min_sing=None
print(A)
print(la.cond(A))

# ## generating ill-conditioned matrix A
# lo = -1000  # low range of random integers
# hi = 1000   # high range of random integers
# n = 100     # dimension (square) of A
# line = [random.randint(lo,hi) for i in range(n)]
# A = np.vander(line)

## generate true solution (int)
# x_true = [random.randint(lo,hi) for i in range(n)]

## generate true solution (R)
x_true = [np.random.normal() for i in range(n)]

## generate data
b = np.dot(A,x_true)

## initial (random) guess
x_init = [np.random.normal() for i in range(n)]

## compute ||Ax - b||
def linear_diff(x,A,b):
    return la.norm(np.dot(A,x)-b)

## various methods for minimizing ||Ax - b|| based on x_init
opts = ['Nelder-Mead','CG','BFGS','L-BFGS-B']

for opt in opts:
    res = sp.optimize.minimize(linear_diff,x_init,args=(A,b),method=opt)
    print('')
    print('-----------------------------------------------------------')
    print('method ',opt, 'resulted in convergence',res.success)
    print('message: ',res.message)
    print('')