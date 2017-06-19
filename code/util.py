import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def mat_from_cond(cond_num, m=50, n=50, min_sing=None):
    """
    Generates an (m x n) matrix with the specified condition number.

    Args:
        (int)   cond_num:   Desired condition number.
        (int)          m:   Desired number of rows.
        (int)          n:   Desired number of columns.
        (float) min_sing:   Desired minimum singular value. Max singular value
                                will equal (cond_num * min_sing). 
    """
    if min_sing == None:
        min_sing = abs(np.random.randn())

    max_sing = min_sing * float(cond_num)

    sing_vals = np.array(sorted([np.random.uniform(low=min_sing, high=max_sing) for _ in range(min(m,n)-2)] + [min_sing, max_sing], reverse=True))


    A = np.random.randn(m, n)
    u,_,v = la.svd(A, full_matrices=True)

    s = np.zeros((m,n))
    for i in range(len(sing_vals)):
        s[i][i] = sing_vals[i]


    return np.dot(u, np.dot(s, v))
