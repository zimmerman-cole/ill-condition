import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util, optimize, test_solvers
import time, datetime
import os

## setup problem
n = 10
# cond_num = 25
# A = util.psd_from_cond(cond_num,n)
# x_true = np.random.randn(n)
# x_0 = np.random.randn(n)
# b = np.dot(A,x_true)

A = np.random.randn(n,n)
U,D,V = la.svd(A)
# print(D)
# can have negative singular values, but by convention, positive


## ITERATIVE REFINEMENT TESTING
# t = test_solvers.Tester()
# t.fit(n_sims=10, cond_num=10**5, m=1000, n=1000, p_xax=0, p_comp=0)
# t.gen_data()
# s1 = "GradientDescentSolver"
# s2 = "ConjugateGradientsSolver"
# s3 = "IterativeRefinementGeneralSolver"
# t.test_spsd(s1)
# t.test_spsd(s2)
# t.test_spsd(solver=s3,intermediate_solver="DirectInverseSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="ConjugateGradientsSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="GradientDescentSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="DirectInverseSolver",intermediate_iter=5,intermediate_continuation=False)
# t.test_spsd(solver=s3,intermediate_solver="ConjugateGradientsSolver",intermediate_iter=5,intermediate_continuation=False)
# t.test_spsd(solver=s3,intermediate_solver="GradientDescentSolver",intermediate_iter=5,intermediate_continuation=False)


## SPECTRUM SHAPE TESTING with multiple methods
n = 50
cond_num = 100
n_sims = 3
t = "decay"
fig = plt.figure("residuals")
ax = plt.subplot(111)

max_iter = 500

# gds = optimize.GradientDescentSolver(A=A,b=b,full_output=True)
# x_gds,i_gds,resids_gds,errs_gds = gds.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
# ax.plot(range(len(resids_gds)), resids_gds, label="gds")

for sim in range(n_sims):
    if t == "decay":
        A = util.decaying_psd(cond_num=cond_num, n=n, min_sing=None)
    else:
        A = util.psd_from_cond(cond_num,n)
    x_true = np.random.randn(n)
    x_0 = np.random.randn(n)
    b = np.dot(A,x_true)

    cgs = optimize.ConjugateGradientsSolver(A=A,b=b,full_output=True)
    x_cgs,i_cgs,resids_cgs,errs_cgs = cgs.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
    resids_cgs = [x[0] for x in resids_cgs]
    ax.plot(range(len(resids_cgs)), resids_cgs, label="cgs")

    ir_dis = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.DirectInverseSolver"), \
                                                       intermediate_iter = 100, \
                                                       intermediate_continuation = True)
    x_ir_dis,i_ir_dis,resids_ir_dis,errs_ir_dis = ir_dis.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
    resids_ir_dis = [x[0] for x in resids_ir_dis]
    ax.plot(range(len(resids_ir_dis)), resids_ir_dis, label="ir_dis")

    ir_cgs = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.ConjugateGradientsSolver"), \
                                                       intermediate_iter = 100, \
                                                       intermediate_continuation = True)
    x_ir_cgs,i_ir_cgs,resids_ir_cgs,errs_ir_cgs = ir_cgs.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
    resids_ir_cgs = [x[0] for x in resids_ir_cgs]
    ax.plot(range(len(resids_ir_cgs)), resids_ir_cgs, label="ir_cgs")

    ir_dis_fix = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.DirectInverseSolver"), \
                                                       intermediate_iter = 100, \
                                                       intermediate_continuation = False)
    x_ir_dis_fix,i_ir_dis_fix,resids_ir_dis_fix,errs_ir_dis_fix = ir_dis_fix.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
    resids_ir_dis_fix = [x[0] for x in resids_ir_dis_fix]
    ax.plot(range(len(resids_ir_dis_fix)), resids_ir_dis_fix, label="ir_dis_fix")

    ir_cgs_fix = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.ConjugateGradientsSolver"), \
                                                       intermediate_iter = 100, \
                                                       intermediate_continuation = False)
    x_ir_cgs_fix,i_ir_cgs_fix,resids_ir_cgs_fix,errs_ir_cgs_fix = ir_cgs_fix.solve(x_0=x_0,x_true=x_true,max_iter=max_iter)
    resids_ir_cgs_fix = [x[0] for x in resids_ir_cgs_fix]
    ax.plot(range(len(resids_ir_cgs_fix)), resids_ir_cgs_fix, label="ir_cgs_fix")


plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("||b-Ax||")
plt.title('Solver Tests')
ax.legend(prop={'size':5})
plt.show()
fname = "../test_results/Spectrum/resids_"+t+".png"
fig.savefig(fname)
