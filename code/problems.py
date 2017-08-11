import numpy as np
import scipy.sparse as sps
import numpy.linalg as la
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
import time, traceback, sys

import util, optimize, template
import tomo2D.blur_1d as blur_1d
import tomo2D.blur_2d as blur_2d
import tomo2D.drt as drt

## =========================================
## TODO: template.py, gen_instance_1d naming
## =========================================

class Problem:
    """
    Parent class for problem types.

    Args:
        - prob     :    'b' = blur, 'x' = x-ray
        - n_1      :    number of rows of image
        - n_2      :    number of cols of image (for 2-d problems)
        - m        :    number of x-rays for x-ray problem
        - k        :    ROI size
        - r        :    ROI row (for 2-d problems)
        - lam      :    regularization parameter
        - B        :    regularization matrix
        - ESI      :    True = generates equiv symm indef representation
        - ESIN     :    True = generates equiv symm indef NORMAL eqn representation
        - dir_soln :    True = generates direct inverse Hotelling template
    """

    def __init__(   self, prob=None, dim=None, \
                    n_1=None, n_2=None, m=None, \
                    k=None, r=None, lam=None, B=None, \
                    ESI=True, ESI_N=True, \
                    dir_soln=True,
                    **kwargs
                ):

        self.prob, self.dim = prob, dim
        self.n_1, self.n_2, self.m = n_1, n_2, m
        self.k, self.r = k, r
        self.lam, self.B = lam, B
        self.ESI, self.ESIN = ESI, ESIN
        self.dir_soln = dir_soln

        if self.dim is None:
            if self.n_2 is None:
                self.dim = 1
                self.n = self.n_1
            else:
                self.dim = 2
                self.n = self.n_1 * self.n_2

        if self.m is None:
            if self.prob == 'b':
                self.m = self.n
            else:
                print('must specify `m` for x-ray problem')
                sys.exit(0)

        if bool(kwargs):
            print('=============== ??? ===============================')
            print('Solver constructor received unexpected arguments:')
            print(kwargs)
            print('Ignoring them and continuing anyways')
            print('===================================================')

    def create_problem(self, **kwargs):
        """
        kwargs:
            + specify either of:
                - plot    :   plot the original image
                - levels  :   number of level changes in original image

            + specify all of:
                - sparse  :   `True` or `False`
                - K_diag  :   diagonal for data covariance matrix Kb
                - sigma   :   sd for Gaussian blur
                - t       :   pixels for discretized Gaussian blur
        """

        ## generate image f ----------------------------------------------------
        if self.dim == 1:
            if 'plot' in kwargs:
                plot = kwargs['plot']
                self.f = blur_1d.gen_f(self.n, plot=plot)
            else:
                self.f = blur_1d.gen_f(self.n)
        elif self.dim == 2:
            if 'levels' in kwargs:
                levels = kwargs['levels']
                if 'plot' in kwargs:
                    plot = kwargs['plot']
                    self.f = blur_2d.gen_f_rect(self.n_1, self.n_2, levels=levels,plot=plot)
                else:
                    self.f = blur_2d.gen_f_rect(self.n_1, self.n_2, levels=levels)
            else:
                if 'plot' in kwargs:
                    plot = kwargs['plot']
                    self.f = blur_2d.gen_f_rect(self.n_1, self.n_2, plot=plot)
                else:
                    self.f = blur_2d.gen_f_rect(self.n_1, self.n_2)
        else:
            print('dim > 2 not implemented yet')
            sys.exit(0)


        ## vectorize image to generate true signal -----------------------------
        self.sx = self.f.flatten("F").reshape(self.n,1)

        ## generate Kb and operators X & M -------------------------------------
        if 'K_diag' and 'sigma' and 't' and 'sparse' in kwargs:
            self.K_diag = kwargs['K_diag']
            self.sigma = kwargs['sigma']
            self.t = kwargs['t']
            self.sparse = kwargs['sparse']
        else:
            print('must specify `K_diag`, `sigma`, `t`, and `sparse`')
            sys.exit(0)

        if self.dim == 1:
            if prob == 'b':
                self.Kb, self.X, self.M = util.gen_instance_1d_blur(m=self.m, n=self.n, k=self.k, \
                                                                    K_diag=self.K_diag, \
                                                                    sigma=self.sigma, t=self.t, \
                                                                    sparse=self.sparse)
            elif prob == 'x':
                self.Kb, self.X, self.M = util.gen_instance_1d_xray(m=self.m, n=self.n, k=self.k, \
                                                                    K_diag=self.K_diag, \
                                                                    sparse=self.sparse)
            else:
                print('problem type not supported, choose `b`-blur or `x`-xray')
        elif self.dim == 2:
            if prob = 'b':
                self.Kb, self.X, self.M = util.gen_instance_2d_blur(m=self.m, n_1=self.n_1, n_2=self.n_1, \
                                                                    ri=self.r, k=self.k,
                                                                    K_diag=self.K_diag, \
                                                                    sigma=self.sigma, t=self.t, \
                                                                    sparse=self.sparse)
            elif prob = 'x':
                self.Kb, self.X, self.M = util.gen_instance_2d_xray(m=self.m, n_1=self.n_1, n_2=self.n_1, \
                                                                    ri=self.r, k=self.k,
                                                                    K_diag=self.K_diag, \
                                                                    sparse=self.sparse)
            else:
                print('problem type not supported, choose `b`-blur or `x`-xray')
        else:
            print('dim > 2 not implemented yet')
            sys.exit(0)

        ## compute data signal -------------------------------------------------
        self.sb = X.dot(self.sx)

        ## adjust regularization if empty --------------------------------------
        if self.B is None: B = sps.eye(n)

        ## generate equivalent symmetric system (ESI) --------------------------
        if self.ESI:
            self.ESI_A, self.ESI_b = util.gen_ESI_system(   X=self.X, Kb=self.Kb, B=self.B, \
                                                            M=self.M, lam=self.lam, sb=self.sb  )

        ## generate ESI^T ESI normal equations ---------------------------------
        if self.ESIN:
            if not self.ESI:
                self.ESI = True
                self.ESI_A, self.ESI_b = util.gen_ESI_system(   X=self.X, Kb=self.Kb, B=self.B, \
                                                                M=self.M, lam=self.lam, sb=self.sb  )
            self.ESIN_A = self.ESI_A.T.dot(self.ESI_A)
            self.ESIN_b = self.ESI_A.T.dot(self.ESI_b)

        ## generate direct solve Hotelling Template (small problems) -----------
        if self.dir_soln:
            self.R_direct = util.direct_rxn(X=self.X, lam=self.lam)
            self.w_direct,_,_ = util.direct_solve(Kb=self.Kb, R=self.R_direct, M=self.M, sb=self.sb)
