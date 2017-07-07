import numpy as np
import numpy.linalg as la
import optimize

# TODO: Max per each column/row diagonal preconditioning
# TODO: Other preconditioning

class Preconditioner:

    def __init__(self):
        pass


class CGPreconditioner:

    def __init__(self, cg_solver, M):
        assert type(cg_solver) == optimize.ConjugateGradientsSolver
        cg_solver._check_ready()
        self.solver = cg_solver

        assert isinstance(M, np.ndarray)
        self.M = M


def jacobi_pre(A):
    """
    Jacobi preconditioner. P = diag(A).

    Works best for diagonally dominant matrices.
    """
    return np.diag(np.diag(A))

def max_diag(A):
    """
    Returned preconditioner is a diagonal matrix, each diagonal element of
        which equals the max-size element of its corresponding column in A.
    """
    P = np.zeros_like(A)
    min_dim = min(len(A), len(A.T))

    for i in range(min_dim):
        P[i, i] = max(A[:, i])

    return P
