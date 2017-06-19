import numpy as np
import numpy.linalg as la
import scipy as sp
import random 

from scipy.optimize import minimize

# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
# res = minimize(rosen, x0, method='Nelder-Mead')
# res.x
# print(rosen(x0))

lo = -1000
hi = 1000
n = 100
line = [random.randint(lo,hi) for i in range(n)]
A = np.vander(line)

print(la.cond(A))
print(A)

x_true = [random.randint(lo,hi) for i in range(n)]
b = np.dot(A,x_true)

x_init = [np.random.normal() for i in range(n)]

def linear_diff(x_init,A,b,normtype='fro'):
    return la.norm(b-np.dot(A,x))

print(linear_diff((A,x_init,b)))

res = sp.optimize.minimize(fun=linear_diff,x0=x_init,args=[A,b,'fro'],method='Nelder-Mead')
print(res.x)