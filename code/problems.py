import numpy as np
import scipy.sparse as sps
import numpy.linalg as la
import scipy.sparse.linalg as spsla
import matplotlib.pyplot as plt
import time, traceback, sys

import util, optimize
import tomo1D.blur_1d as blur_1d
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
        - ESI3     :    True = generates expanded ESI 3x3 system per Sean's notes
        - dir_soln :    True = generates direct inverse Hotelling template
    """

    def __init__(   self, prob=None, dim=None, \
                    n_1=None, n_2=None, m=None, \
                    k=None, r=None, lam=None, B=None, \
                    ESI=True, ESIN=True, ESI3=True, \
                    dir_soln=True,
                    **kwargs
                ):

        self.prob, self.dim = prob, dim
        self.n_1, self.n_2, self.m = n_1, n_2, m
        self.k, self.r = k, r
        self.lam, self.B = lam, B
        self.ESI, self.ESIN, self.ESI3 = ESI, ESIN, ESI3
        self.dir_soln = dir_soln

        if self.n_2 is not None:
            self.n = self.n_1 * self.n_2
        else:
            self.n = self.n_1

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
        if self.r is None:
            self.r = int(self.n_1/2)

        if bool(kwargs):
            print('=============== ??? ===============================')
            print('Solver constructor received unexpected arguments:')
            print(kwargs)
            print('Ignoring them and continuing anyways')
            print('===================================================')

    def __str__(self):
        l0 = '=================== setup ====================\n'
        l1 = '(n_1, n_2, m) = (' + str(self.n_1) +', ' + str(self.n_2) + ', ' + str(self.m) + ')\n'
        if self.prob == 'b':
            l2 = 'problem       = ' + str(self.dim) + 'D Blur\n'
        elif self.prob == 'x':
            l2 = 'problem       = ' + str(self.dim) + 'D X-Ray\n'
        else:
            print('problem type not supported, choose `b`-blur or `x`-xray')
            sys.exit(0)
        l3 = 'lam           = ' + str(self.lam) +'\n'
        l4 = 'B             = ' + str(type(self.B)) +'\n'
        l5 = 'ROI pixels    = ' + str(self.k) +'\n'
        l6 = 'ROI row       = ' + str(self.r) +'\n'

        return l0+l1+l2+l3+l4+l5+l6

    def __repr__(self):
        return self.__str__()

    def _set_inputs(self, **kwargs):
        if 'K_diag' and 'sigma' and 't' and 'sparse' in kwargs:
            self.K_diag = kwargs['K_diag']
            self.sigma = kwargs['sigma']
            self.t = kwargs['t']
            self.sparse = kwargs['sparse']
        else:
            print('must specify all of `K_diag`, `sigma`, `t`, and `sparse`')
            raise

    def _set_image(self, **kwargs):
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

    def _set_operators(self, **kwargs):
        ## generate Kb and operators X & M -------------------------------------
        if self.dim == 1:
            if self.prob == 'b':
                self.Kb, self.X, self.M = util.gen_instance_1d_blur(m=self.m, n=self.n, k=self.k, \
                                                                    K_diag=self.K_diag, \
                                                                    sigma=self.sigma, t=self.t, \
                                                                    sparse=self.sparse)
            elif self.prob == 'x':
                print('`x`-xray only allowed for 2D')
                sys.exit(0)
            else:
                print('problem type not supported, choose `b`-blur or `x`-xray')
                sys.exit(0)
        elif self.dim == 2:
            if self.prob == 'b':
                self.Kb, self.X, self.M = util.gen_instance_2d_blur(m=self.m, n_1=self.n_1, n_2=self.n_2, \
                                                                    ri=self.r, k=self.k,
                                                                    K_diag=self.K_diag, \
                                                                    sigma=self.sigma, t=self.t, \
                                                                    sparse=self.sparse)
            elif self.prob == 'x':
                self.Kb, self.X, self.M = util.gen_instance_2d_xray(m=self.m, n_1=self.n_1, n_2=self.n_2, \
                                                                    ri=self.r, k=self.k,
                                                                    K_diag=self.K_diag, \
                                                                    sparse=self.sparse)
            else:
                print('problem type not supported, choose `b`-blur or `x`-xray')
                sys.exit(0)
        else:
            print('dim > 2 not implemented yet')
            sys.exit(0)

        ## generate B regularization if empty ----------------------------------
        if self.B is None: self.B = sps.eye(self.n)

    def _set_systems(self, **kwargs):
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

        ## generate ESI3 equations ---------------------------------------------
        if self.ESI3:
            self.ESI3_A, self.ESI3_b = util.gen_ESI3_system(   X=self.X, Kb=self.Kb, B=self.B, \
                                                            M=self.M, lam=self.lam, sb=self.sb  )

    def _set_direct(self, **kwargs):
        ## generate direct solve Hotelling Template (small problems) -----------
        if self.dir_soln:
            self.R_direct = util.direct_rxn(X=self.X, lam=self.lam)
            self.w_direct,_,_ = util.direct_solve(Kb=self.Kb, R=self.R_direct, M=self.M, sb=self.sb)

    def create_problem(self, **kwargs):
        """
        kwargs:
            + specify ANY/NONE of:
                - plot    :   plot the original image
                - levels  :   number of level changes in original image

            + specify ALL of:
                - sparse  :   `True` or `False`
                - K_diag  :   diagonal for data covariance matrix Kb
                - sigma   :   sd for Gaussian blur
                - t       :   pixels for discretized Gaussian blur
        """
        ## set attributes ------------------------------------------------------
        self._set_inputs(**kwargs)
        self._set_image(**kwargs)
        self._set_operators(**kwargs)

        ## set data signal -----------------------------------------------------
        self.sb = self.X.dot(self.sx)

        ## set direct ESI systems ----------------------------------------------
        if self.ESI or self.ESIN or self.ESI3:
            self._set_systems(**kwargs)

        ## set direct solution -------------------------------------------------
        if self.dir_soln:
            self._set_direct(**kwargs)


    def summarize(self):
        print(self.__repr__())
        print('================== contents ==================')
        print('K_diag        = ' + str(self.K_diag[0:5]) + '...' +  str(self.K_diag[-6:-1]))
        print('sigma         = ' + str(self.sigma))
        print('t             = ' + str(self.t))
        print('ESI?          = ' + str(self.ESI))
        print('ESIN?         = ' + str(self.ESIN))
        print('ESI3?         = ' + str(self.ESI3))
        print('direct?       = ' + str(self.dir_soln))
        print('================= dimensions ==================')
        print('Kb shape      = ' + str(self.Kb.shape))
        print('X shape       = ' + str(self.X.shape))
        print('M shape       = ' + str(self.M.shape))
        if self.B is None:
            print('B shape       = None')
        else:
            print('B shape       = ' + str(self.B.shape))
        print('sx shape      = ' + str(self.sx.shape))
        print('sb shape      = ' + str(self.sb.shape))
        print('============= system dimensions ===============')
        print('ESI_A shape   = ' + str(self.ESI_A.shape))
        print('ESI_b shape   = ' + str(self.ESI_b.shape))
        print('ESIN_A shape  = ' + str(self.ESIN_A.shape))
        print('ESIN_b shape  = ' + str(self.ESIN_b.shape))
        print('ESI3_A shape  = ' + str(self.ESI3_A.shape))
        print('ESI3_b shape  = ' + str(self.ESI3_b.shape))

if __name__ == '__main__':
    ## 1D BLUR
    p_1d_blur = Problem(prob='b', dim=1, \
                        n_1=200, n_2=None, m=None, \
                        k=50, r=None, lam=100, B=None, \
                        ESI=True, ESIN=True, \
                        dir_soln=True)
    p_1d_blur.create_problem(K_diag=np.ones(p_1d_blur.n), sigma=3, t=10, sparse=True)
    p_1d_blur.summarize()
    print('')

    # ## 1D (implicit) BLUR
    # p_1d_blur = Problem(prob='b', dim=None, \
    #                     n_1=200, n_2=None, m=None, \
    #                     k=50, r=None, lam=100, B=None, \
    #                     ESI=True, ESIN=True, \
    #                     dir_soln=True)
    # p_1d_blur.create_problem(K_diag=np.ones(p_1d_blur.n), sigma=3, t=10, sparse=True)
    # p_1d_blur.summarize()
    # print('')

    ## 2D BLUR
    p_2d_blur = Problem(prob='b', dim=2, \
                        n_1=20, n_2=50, m=None, \
                        k=50, r=None, lam=100, B=None, \
                        ESI=True, ESIN=True, \
                        dir_soln=True)
    p_2d_blur.create_problem(K_diag=np.ones(p_2d_blur.m), sigma=3, t=10, sparse=True)
    p_2d_blur.summarize()
    print('')


    ## 2D XRAY
    p_2d_xray = Problem(prob='x', dim=2, \
                        n_1=20, n_2=50, m=75, \
                        k=50, r=None, lam=100, B=None, \
                        ESI=True, ESIN=True, \
                        dir_soln=True)
    p_2d_xray.create_problem(K_diag=np.ones(p_2d_xray.m), sigma=None, t=None, sparse=True)
    p_2d_xray.summarize()
    print('')


    ## 1D XRAY (not allowed)
    p_1d_xray = Problem(prob='x', dim=1, \
                        n_1=200, n_2=None, m=20, \
                        k=50, r=None, lam=100, B=None, \
                        ESI=True, ESIN=True, \
                        dir_soln=True)
    p_1d_xray.create_problem(K_diag=np.ones(p_1d_xray.n), sigma=None, t=None, sparse=True)
    p_1d_xray.summarize()
    print('')
